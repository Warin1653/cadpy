import os
import geopandas as gpd
import pandas as pd
import rtree
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


class cadpy:
    """
    A class for processing and manipulating cadastral data.
    
    Provides methods for spatial operations including:
    - Polygon simplification
    - Merging operations
    - Spatial joins
    - Dask-based spatial operations
    """
    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        pin: str,
        proj_crs: str
    ):
        
        """
        Initialize the CadPy processor
        
        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame of the cadastral data
            pin (str): Column name of parcel identification number
            proj_crs (str): Projected CRS for geoprocessing
        """
        self.gdf = gdf
        self.pin = pin
        self.proj_crs = proj_crs   

    
    def _create_spatial_index(self, gdf):
        """Create a spatial index for a GeoDataFrame to optimize spatial queries."""
        idx = rtree.index.Index()
        for i, geometry in enumerate(gdf.geometry):
            idx.insert(i, geometry.bounds)
        return idx
  

    def _explode(self, gdf):
        """Split multi-part polygons into single parts"""
        result_gdf = gdf.copy()

        # Explode multiparts into single parts
        result_gdf = result_gdf.explode(index_parts=False)

        # Generate unique part_index for each of the exploded parts
        result_gdf["Part_Index"] = result_gdf.groupby(self.pin).cumcount() + 1 

        # Conditionally append suffix only if Part_Index > 1 for the exploded parts
        result_gdf[self.pin] = result_gdf[self.pin].astype(str) + result_gdf["Part_Index"].apply(lambda x: f"_{x}" if x > 1 else "")

        # Reset index
        result_gdf = result_gdf.reset_index(drop=False)

        return result_gdf


    def merge_by_attribute(self, gdf, attribute):
        """
        Merge intersecting polygons that share the same attribute value.

        Parameters:
            - gdf: GeoDataFrame containing cadastre polygons
            - attribute: Column name to use for grouping polygons for merging

        Returns:
            - GeoDataFrame with merged polygons
        """
        result_gdf = gdf.copy()

        # Check if the attribute exists in the dataframe
        if attribute not in result_gdf.columns:
            print(f"Attribute '{attribute}' not found in the GeoDataFrame.")
            return gdf

        # Create spatial index
        spatial_index = self._create_spatial_index(result_gdf)

        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for idx, row in result_gdf.iterrows():
            # Skip if this polygon has already been marked for removal
            if idx in indices_to_drop:
                continue

            # Find potential matches using the spatial index
            intersecting_indices = list(spatial_index.intersection(row.geometry.bounds))
            potential_matches = result_gdf.iloc[intersecting_indices]

            # Filter for polygons with the same attribute value that intersect the current polygon
            potential_matches = potential_matches[
                (potential_matches[attribute] == row[attribute]) & 
                (potential_matches.index != idx) & 
                (potential_matches.geometry.intersects(row.geometry))
            ]

            if not potential_matches.empty:
                # Create a merged polygon from the current polygon and all matching polygons
                polygons_to_merge = [result_gdf.loc[idx, "geometry"]] + potential_matches.geometry.tolist()

                try:
                    # Attempt to merge the polygons
                    merged_geometry = unary_union(polygons_to_merge)

                    # Update the current polygon's geometry with the merged geometry
                    result_gdf.at[idx, "geometry"] = merged_geometry

                    # Mark the matching polygons for removal
                    indices_to_drop.update(potential_matches.index)

                except Exception as e:
                    # If merge fails, continue with the next polygon
                    print(f"Merge failed for index {idx}- {e}")
                    continue

        # Remove all merged polygons
        if indices_to_drop:
            result_gdf = result_gdf.drop(index=indices_to_drop).reset_index(drop=True)

        return result_gdf


    def _merge_by_filled_polygon(self, gdf):
        """
        Merge building footprints to the larger polygon using the filled polygon method.

        Parameters:
            - gdf: GeoDataFrame containing cadastre polygons

        Returns:
            - GeoDataFrame with merged polygons
        """
        # Create a copy to avoid modifying the input
        result_gdf = gdf.copy()

        # Create spatial index
        spatial_index = self._create_spatial_index(result_gdf)

        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for index, row in result_gdf.iterrows():
            # Skip if this polygon has already been marked for removal
            if index in indices_to_drop:
                continue

            # Handle Polygon and MultiPolygon separately
            if isinstance(row.geometry, Polygon):
                # For a single Polygon, create a "filled" version without holes
                filled_polygon = Polygon(row.geometry.exterior)
            elif isinstance(row.geometry, MultiPolygon):
                # For MultiPolygon, create a "filled" MultiPolygon without holes
                filled_polygon = MultiPolygon([Polygon(poly.exterior) for poly in row.geometry.geoms])

            # Get the polygons that are within the filled
            intersecting_indices = list(spatial_index.intersection(filled_polygon.bounds))
            potential_matches = result_gdf.iloc[intersecting_indices]
            potential_matches = potential_matches[
                potential_matches.buffer(-0.1).geometry.within(filled_polygon.buffer(0.1)) & 
                (potential_matches.index != index)
            ]

            if not potential_matches.empty:
                # Create a merged polygon from the current polygon and all matching polygons
                polygons_to_merge = [row.geometry] + potential_matches.geometry.tolist()

                try:
                    # Attempt to merge the polygons
                    merged_geometry = unary_union(polygons_to_merge)

                    # Update the current polygon's geometry with the merged geometry
                    result_gdf.at[index, "geometry"] = merged_geometry

                    # Get pin information from the child polygons
                    pin_list = potential_matches[self.pin].tolist()

                    # Store the pin information of the child polygons in a new column
                    result_gdf.at[index, "merged_pin"] = str(pin_list)

                    # Mark the matching polygons for removal
                    indices_to_drop.update(potential_matches.index)

                except Exception as e:
                    # If merge fails, continue with the next polygon
                    print(f"merge failed for {row.PFI}- {e}")
                    continue

        # Remove all merged polygons
        if indices_to_drop:
            result_gdf = result_gdf.drop(index=indices_to_drop).reset_index(drop=True)

        return result_gdf

    def simplify_cadastre(self,
                          merge_by_attribute = False,
                          attribute = None):
        """
        Simplify/flatten any overlapping parcels and dissolve enclosed parcels.

        Parameters:
            - gdf: Input GeoDataFrame containing cadastre polygons.
            - pin: Parcel identification number column
            - proj_crs: Projected CRS for calculating area.

        Returns:
            - Flattened GeoDataFrame with retained attributes.
        """

        # Make a copy of the input GeoDataFrame
        result_gdf = self.gdf.copy()
        
        # Convert pin column to string
        result_gdf[self.pin] = result_gdf[self.pin].astype(int).astype(str)

        # Explode into single parts
        result_gdf = self._explode(result_gdf)
        
        # Merge by attribute if flagged
        if merge_by_attribute:
            result_gdf = self.merge_by_attribute(result_gdf, attribute)

        # Transform into projected CRS
        result_gdf = result_gdf.to_crs(self.proj_crs)

        # Merge building footprints to the larger polygon using the filled polygon method
        result_gdf = self._merge_by_filled_polygon(result_gdf)

        return result_gdf