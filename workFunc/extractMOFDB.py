import os
import json
from mofdb_client import fetch
from tqdm.auto import tqdm


# Maps common gas keys (formulas, short names, or the mofdb adsorbate name
# itself) to the lowercased adsorbate name string used internally by mofdb.
# Confirmed against real mofdb records: CO2, N2, H2, CH4, Kr, Xe.
# He/Ne/Ar are inferred from the same naming pattern (concatenated word,
# no spaces) but not directly verified against a live record -- if you use
# them and get zero matches, check `ad.name` on a fetched isotherm's
# adsorbates to confirm the exact string mofdb uses.
GAS_ALIASES = {
    "co2": "carbondioxide", "carbon dioxide": "carbondioxide", "carbondioxide": "carbondioxide",
    "n2": "nitrogen", "nitrogen": "nitrogen",
    "h2": "hydrogen", "hydrogen": "hydrogen",
    "ch4": "methane", "methane": "methane",
    "xe": "xenon", "xenon": "xenon",
    "kr": "krypton", "krypton": "krypton",
    "ar": "argon", "argon": "argon",       # unverified
    "ne": "neon", "neon": "neon",          # unverified
    "he": "helium", "helium": "helium",    # unverified
}


def resolve_gas_name(gas):
    """
    Normalize a user-supplied gas identifier to the lowercased adsorbate
    name string used internally by mofdb.

    Args:
        gas (str): A gas formula (e.g. "CO2"), common name (e.g.
            "carbon dioxide"), or the mofdb adsorbate name itself (e.g.
            "CarbonDioxide"). Case-insensitive.

    Returns:
        str: The lowercased canonical adsorbate name to compare against
        `ad.name.lower()` / `sp.name.lower()` when scanning isotherms.

    Raises:
        ValueError: If `gas` isn't a recognized alias. The error lists
            the known aliases to help pick a valid value.
    """
    key = gas.strip().lower()
    if key not in GAS_ALIASES:
        raise ValueError(
            f"Unrecognized gas '{gas}'. Known aliases: {sorted(GAS_ALIASES)}"
        )
    return GAS_ALIASES[key]


class MOFDatasetBuilder:
    """
    Builds a local dataset of Metal-Organic Framework (MOF) structures and
    methane adsorption isotherm data pulled from mofdb.

    For each MOF returned by the mofdb API, this class:
      - saves the MOF's CIF structure file to `output_dir`
      - extracts every methane adsorption data point across all of the
        MOF's isotherms (temperature, pressure, adsorption capacity,
        composition, source DOI, etc.)
      - writes all extracted records (one row per MOF-isotherm-pressure
        point) to a single JSON file at `output_json`

    Attributes:
        output_dir (str): Directory where individual .cif files are saved.
        output_json (str): Path to the JSON file where extracted records
            are saved.
    """

    def __init__(self, output_dir="mofs", output_json="Dataset/mofs_data.json"):
        """
        Initialize the builder and ensure output directories exist.

        Args:
            output_dir (str): Directory to save individual MOF .cif files.
                Defaults to "mofs".
            output_json (str): Path (including filename) where the final
                JSON dataset will be written. Defaults to
                "Dataset/mofs_data.json". The parent directory is created
                if it doesn't already exist.
        """
        os.makedirs("Dataset", exist_ok=True)
        self.output_dir = output_dir
        self.output_json = output_json

        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_mofs(self, vf_min=0.5, vf_max=0.99, loading_unit="mmol/g",
                   pressure_unit="atm", limit=None):
        """
        Query mofdb for MOFs matching the given void fraction range.

        This is a thin wrapper around `mofdb_client.fetch`. The result is
        a lazy generator, so it should be iterated with a `for` loop
        rather than materialized into a list (doing so would download the
        entire result set into memory before processing starts).

        Args:
            vf_min (float): Minimum void fraction (inclusive). Defaults to 0.5.
            vf_max (float): Maximum void fraction (inclusive). Defaults to 0.99.
            loading_unit (str): Unit to convert all isotherm adsorption
                (loading) values to. Defaults to "mmol/g".
            pressure_unit (str): Unit to convert all isotherm pressure
                values to. Defaults to "atm".
            limit (int or None): Maximum number of MOFs to fetch. If
                None, fetches all MOFs matching the filters (no limit).
                Defaults to None.

        Returns:
            Iterator of MOF objects as returned by mofdb_client.fetch.
        """
        mofs = fetch(vf_min=vf_min, vf_max=vf_max, loading_unit=loading_unit,
                      pressure_unit=pressure_unit, limit=limit)
        return mofs

    def save_cif(self, mof):
        """
        Write a MOF's CIF structure to disk.

        Args:
            mof: A MOF object (from mofdb_client) with `.name` and `.cif`
                attributes.

        Side Effects:
            Creates a file at `{self.output_dir}/{mof.name}.cif`.
        """
        path = os.path.join(self.output_dir, f"{mof.name}.cif")
        with open(path, "w") as f:
            f.write(mof.cif)

    def extract_gas_points(self, mof, gas="methane"):
        """
        Extract every adsorption data point for a given gas from a MOF's
        isotherms.

        Scans all isotherms belonging to the MOF, keeps only those that
        include the target gas as an adsorbate, and for each pressure
        point in those isotherms pulls out the gas-specific adsorption
        value.

        Args:
            mof: A MOF object with an `.isotherms` attribute.
            gas (str): Gas to extract, given as a formula (e.g. "CO2"),
                common name (e.g. "carbon dioxide"), or mofdb adsorbate
                name (e.g. "CarbonDioxide"). Case-insensitive. Defaults
                to "methane". See `GAS_ALIASES` for all recognized
                values -- CO2, N2, H2, CH4, Kr, and Xe are confirmed
                against real mofdb records; He, Ne, and Ar are supported
                but unverified (double check `ad.name` on live data if
                you rely on these and get no matches).

        Returns:
            list[dict]: One dict per matching data point, each containing:
                - isotherm_id: ID of the source isotherm
                - temperature_K: Isotherm temperature (Kelvin)
                - pressure_atm: Pressure at this data point (atm)
                - adsorption_mmol_g: Adsorption capacity for `gas` (mmol/g)
                - composition: `gas`'s mole/weight fraction at this point
                - composition_type: What `composition` is measured in
                  (e.g. "molefraction" or "wt%")
                - doi: DOI of the source publication for this isotherm
                - gas: The canonical gas name matched (lowercased)

        Raises:
            ValueError: If `gas` isn't a recognized alias.
        """
        gas_name = resolve_gas_name(gas)
        gas_points = []

        for iso in mof.isotherms:
            ads_names = [ad.name.lower() for ad in iso.adsorbates]

            if gas_name not in ads_names:
                continue

            for point in iso.isotherm_data:

                match = next((sp for sp in point.species_data
                    if sp.name.lower() == gas_name), None)

                if match is None:
                    continue

                gas_points.append(
                    {
                        "isotherm_id": iso.id,
                        "temperature_K": iso.temperature,
                        "pressure_atm": point.pressure,
                        "adsorption_mmol_g": match.adsorption,
                        "composition": match.composition,
                        "composition_type": iso.compositionType,
                        "doi": iso.DOI,
                        "gas": gas_name,
                    }
                )

        return gas_points

    def build_records(self, mof, gas="methane"):
        """
        Build flat dataset records for a single MOF.

        Combines the MOF's static metadata (structural properties,
        composition, etc.) with each of its adsorption data points for
        the target gas, producing one fully self-contained record per
        (isotherm, pressure) pair. A MOF with no isotherms for `gas`
        contributes zero records.

        Args:
            mof: A MOF object with structural attributes (mofid, mofkey,
                name, void_fraction, surface_area_m2g, surface_area_m2cm3,
                pld, lcd, pxrd, pore_size_distribution, elements) and an
                `.isotherms` attribute.
            gas (str): Gas to extract records for, given as a formula
                (e.g. "CO2"), common name, or mofdb adsorbate name.
                Case-insensitive. Defaults to "methane". See
                `GAS_ALIASES` for recognized values -- CO2, N2, H2, CH4,
                Kr, and Xe are confirmed against real mofdb records; He,
                Ne, and Ar are supported but unverified.

        Returns:
            list[dict]: One record per (isotherm, pressure) data point
            for `gas`, each containing the MOF's metadata merged with:
                - temperature_K, pressure_atm, adsorption_mmol_g,
                  composition, composition_type, doi, isotherm_id, gas

        Raises:
            ValueError: If `gas` isn't a recognized alias.
        """
        gas_name = resolve_gas_name(gas)

        base = {
            "mofid": mof.mofid,
            "mofkey": mof.mofkey,
            "name": mof.name,
            "void_fraction": mof.void_fraction,
            "surface_area_m2g": mof.surface_area_m2g,
            "surface_area_m2cm3": mof.surface_area_m2cm3,
            "pld": mof.pld,
            "lcd": mof.lcd,
            "pxrd": mof.pxrd,
            "pore_size_distribution": mof.pore_size_distribution,
            "elements": [
                {"symbol": el.symbol, "name": el.name}
                for el in mof.elements
            ],
        }

        records = []

        for iso in mof.isotherms:
            if gas_name not in [ad.name.lower() for ad in iso.adsorbates]:
                continue

            for point in iso.isotherm_data:
                match = next(
                    (sp for sp in point.species_data if sp.name.lower() == gas_name),
                    None,
                )

                if match is None:
                    continue

                record = base.copy()
                record.update({
                    "temperature_K": iso.temperature,
                    "pressure_atm": point.pressure,
                    "adsorption_mmol_g": match.adsorption,
                    "composition": match.composition,
                    "composition_type": iso.compositionType,
                    "doi": iso.DOI,
                    "isotherm_id": iso.id,
                    "gas": gas_name,
                })

                records.append(record)

        return records

    def save_json(self, records):
        """
        Write extracted records to the output JSON file.

        Args:
            records (list[dict]): Records to serialize, typically the
                accumulated output of `build_records` across all MOFs.

        Side Effects:
            Overwrites the file at `self.output_json` with pretty-printed
            JSON (indent=2).
        """
        with open(self.output_json, "w") as f:
            json.dump(records, f, indent=2)

    def build_dataset(
        self,
        vf_min=0.5,
        vf_max=0.99,
        loading_unit="mmol/g",
        pressure_unit="atm",
        limit=5,
        gas="methane",
    ):
        """
        Run the full dataset build pipeline end-to-end.

        Fetches MOFs matching the given filters, saves each MOF's CIF
        file, extracts adsorption records for the target gas from each
        MOF's isotherms, and writes the combined records to
        `self.output_json`.

        Args:
            vf_min (float): Minimum void fraction (inclusive). Defaults to 0.5.
            vf_max (float): Maximum void fraction (inclusive). Defaults to 0.99.
            loading_unit (str): Unit to convert all isotherm adsorption
                values to. Defaults to "mmol/g".
            pressure_unit (str): Unit to convert all isotherm pressure
                values to. Defaults to "atm".
            limit (int or None): Maximum number of MOFs to process. If
                None, processes all MOFs matching the filters. Defaults
                to 5.
            gas (str): Gas to extract adsorption records for, given as a
                formula (e.g. "CO2"), common name, or mofdb adsorbate
                name. Case-insensitive. Defaults to "methane". See
                `GAS_ALIASES` for recognized values

        Side Effects:
            - Writes one .cif file per MOF to `self.output_dir`.
            - Writes the full set of extracted records to
              `self.output_json`.
            - Prints a summary line with the number of records saved.

        Raises:
            ValueError: If `gas` isn't a recognized alias (raised via
                `build_records`/`resolve_gas_name`).
        """
        records = []

        mofs = self.fetch_mofs(
            vf_min=vf_min,
            vf_max=vf_max,
            loading_unit=loading_unit,
            pressure_unit=pressure_unit,
            limit=limit,
        )

        for mof in tqdm(mofs, total=limit, desc="Processing MOFs"):
            self.save_cif(mof)
            records.extend(self.build_records(mof, gas=gas))

        self.save_json(records)

        print(f"Saved {len(records)} records to {self.output_json}")