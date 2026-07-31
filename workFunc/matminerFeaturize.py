import os, re, warnings 
import pandas as pd 
from tqdm.auto import tqdm
from pymatgen.core import Structure 
from pathlib import Path
from matminer.featurizers.composition import (
    ElementProperty, TMetalFraction, ValenceOrbital
    )
from matminer.featurizers.structure import (
    DensityFeatures, GlobalSymmetryFeatures
    )

class featurizer:
    """
    Composition and cell level MOF features 
    No pore geometry or metal-node chemistry descriptor
    """
    _BAD = re.compile(r"[^0-9a-zA-Z]+")

    def __init__(self, cif_dir="mofs", prefix="base_", 
                 require_ordered=True, max_sites=None):
        """
        Configure the featurizer.

        Parameters
        ----------
        cif_dir : str
            Directory containing the .cif structure files.
        prefix : str
            String prepended to every generated feature column name.
        require_ordered : bool
            If True, reject structures with partial/disordered occupancy.
        max_sites : int or None
            If set, reject structures with more sites than this (perf guard).
        """
        self.cif_dir = cif_dir 
        self.prefix = prefix 
        self.require_ordered = require_ordered 
        self.max_sites = max_sites 
        self._f = {
                "density":  (DensityFeatures(),        "structure"),
                "symmetry": (GlobalSymmetryFeatures(), "structure"),
                "magpie":   (ElementProperty.from_preset("magpie"), "composition"),
                "tmetal":   (TMetalFraction(),         "composition"),
                "valence":  (ValenceOrbital(),         "composition"),
            }

    def _key(self, k):
        """
        Turn a raw matminer feature label into a valid, prefixed column
        name, e.g. 'packing fraction' -> 'base_packing_fraction',
        'MagpieData mean Number' -> 'base_MagpieData_mean_Number'.
        """
        return self.prefix + self._BAD.sub("_", k).strip("_")

    def _load(self, name):
        """
        Load a single structure by name from cif_dir, applying the
        ordered/max_sites filters. Raises FileNotFoundError or ValueError
        if the structure is missing or fails validation.
        """
        path = os.path.join(self.cif_dir, f"{name}.cif") 
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = Structure.from_file(path)
        if self.require_ordered and not s.is_ordered:
            raise ValueError("disordered / partial occupancy")
        if self.max_sites and len(s) > self.max_sites:
            raise ValueError(f"{len(s)} sites exceeds max_sites")
        return s

    def discover_target(self):
        """
        Scan cif_dir for all .cif files and return a DataFrame with one
        'name' column (the file stems), sorted alphabetically. Used to
        build the initial target list before featurizing.
        """
        names = sorted(p.stem for p in Path(self.cif_dir).glob("*.cif"))
        return pd.DataFrame({"name":names})
        
    def featurize_one(self,name, n_sites=False, volume=False):
        """
        Compute all features for a single structure by name.

        Loads the structure, then runs every featurizer in self._f over
        it (or its composition), collecting results into a flat dict.
        Any per-featurizer failure is caught and recorded in rec["errors"]
        rather than raising, so a single bad structure/featurizer doesn't
        abort a batch run. rec["ok"] is True only if no errors occurred.

        Parameters
        ----------
        name : str
            Structure name (matches a .cif file stem in cif_dir).
        n_sites, volume : bool
            If True, expose 'n_sites'/'volume' as prefixed feature columns
            (included in numeric_matrix output); if False, keep them as
            plain metadata columns.

        Returns
        -------
        dict
            Record with name, ok flag, errors dict, and feature values
            (or just {"name", "ok": False, "errors": {...}} on load failure).
        """
        rec = {"name": name, "ok": False, "errors": {}}
        try:
            s = self._load(name)
        except Exception as e:
            rec["errors"]["load"] = str(e)[:200]
            return rec

        rec.update({
            "formula": s.composition.reduced_formula,
            "has_H": "H" in s.composition.as_dict(),
        })

        # promoted to features only when asked for
        rec[self._key("n_sites") if n_sites else "n_sites"] = len(s)
        rec[self._key("volume") if volume else "volume"] = float(s.volume)

        
        comp = s.composition
        for tag, (obj, kind) in self._f.items():
            try:
                arg = s if kind == "structure" else comp
                vals = obj.featurize(arg)
                rec.update({self._key(k): v
                            for k, v in zip(obj.feature_labels(), vals)})
            except Exception as e:
                rec["errors"][tag] = str(e)[:200]

        rec["ok"] = len(rec["errors"]) == 0
        #True if errors col. are empty
        return rec

    def featurize(self, names, verbose=True, 
                 n_sites=False, volume=False):
        """
        Batch version of featurize_one: runs over an iterable of names
        (or a DataFrame with a 'name' column), shows a tqdm progress bar,
        and returns one DataFrame with one row per structure.

        If verbose, prints a "clean" count summary and a breakdown of
        which error tags occurred most often across the batch.
        """
        if isinstance(names, pd.DataFrame):
            names = names["name"]
        names = tqdm(list(names), desc="baseline")

        df = pd.DataFrame([self.featurize_one(n, n_sites=n_sites,
                                             volume=volume) 
                           for n in names])
        if verbose and "ok" in df:
            print(f"{int(df['ok'].sum())}/{len(df)} clean")
            bad = [k for e in df["errors"] if isinstance(e, dict) for k in e]
            if bad:
                print(pd.Series(bad).value_counts())
        return df

    def attach(self, target_df, feature_df,
               name_col="name",
               drop_failed=True):
        """
        Left-merge computed features onto an arbitrary target DataFrame.

        Parameters
        ----------
        target_df : pd.DataFrame
            The DataFrame to attach features to (e.g. a labels table).
        feature_df : pd.DataFrame
            Output of featurize(); must contain 'name', 'ok', 'errors'.
        name_col : str
            Column in target_df that matches feature_df['name'].
        drop_failed : bool
            If True, exclude rows where ok == False before merging (so
            failed structures end up with NaN feature columns instead of
            stale/partial values).

        Returns
        -------
        pd.DataFrame
            target_df with feature columns merged in (how="left").
        """
        if drop_failed:
            feature_df = feature_df[feature_df["ok"]]
    
        feature_df = feature_df.drop(
            columns=["errors", "ok"],
            errors="ignore"
        )
    
        merged_df = target_df.merge(
            feature_df,
            left_on=name_col,
            right_on="name",
            how="left"
        )
    
        return merged_df


    def numeric_matrix(self, df):
        """
        Reduce a merged DataFrame to a model-ready numeric matrix: keep
        only columns starting with self.prefix, one-hot encode any
        object/string columns (with a NaN indicator column), and coerce
        everything else to numeric (non-numeric values become NaN).
        """
        cols = [c for c in df.columns if c.startswith(self.prefix)]
        sub = df[cols]
        obj = sub.select_dtypes(include="object").columns
        if len(obj):
            sub = pd.get_dummies(sub, columns=list(obj), dummy_na=True)
        return sub.apply(pd.to_numeric, errors="coerce")

if __name__ == "__main__":
    fz = featurizer(cif_dir="mofs", max_sites=3000)
    target = fz.discover_target()

    feature_df = fz.featurize(target["name"])
    merged_df = fz.attach(target, feature_df)
    numeric_df = fz.numeric_matrix(merged_df)

    print(merged_df.shape, "->", numeric_df.shape)