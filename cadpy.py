import os
import geopandas as gpd
import pandas as pd
import rtree
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def _create_spatial_index(gdf):
    """Create a spatial index for a GeoDataFrame to optimize spatial queries."""
    idx = rtree.index.Index()
    for i, geometry in enumerate(gdf.geometry):
        idx.insert(i, geometry.bounds)
    return idx


def flatten_cadastre2(gdf, pin, proj_crs):
    """
    Merge building footprints to the larger polygon using the filled polygon method.
    
    Parameters:
        - gdf: GeoDataFrame containing cadastre polygons
        
    Returns:
        - GeoDataFrame with merged polygons
    """
    
    # Create a set to store indices of polygons to drop
    indices_to_drop = set()
    
    # Create a copy to avoid modifying the input
    result_gdf = gdf.copy()
    
    # Create spatial index
    spatial_index = _create_spatial_index(result_gdf)
    
    # Calculate area of each parcel
    result_gdf = result_gdf.to_crs(proj_crs)
    
    # Initiailize a column to store merged pin
    result_gdf["merged_pin"] = None

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
                pin_list = potential_matches[pin].tolist()

                # Store the pin information of the merged polygons in a new column
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


def merge_by_attribute(gdf, attribute):
    """
    Merge intersecting polygons that share the same attribute value.
    
    Parameters:
        - gdf: GeoDataFrame containing cadastre polygons
        - attribute: Column name to use for grouping polygons for merging
        
    Returns:
        - GeoDataFrame with merged polygons
    """
    # Check if the attribute exists in the dataframe
    if attribute not in gdf.columns:
        print(f"Attribute '{attribute}' not found in the GeoDataFrame. Skipping merge.")
        return gdf
    
    # Create spatial index
    spatial_index = create_spatial_index(gdf)
    
    # Create a set to store indices of polygons to drop
    indices_to_drop = set()

    # Iterate over rows
    for idx, row in gdf.iterrows():
        # Skip if this polygon has already been marked for removal
        if idx in indices_to_drop:
            continue

        # Find potential matches using the spatial index
        intersecting_indices = list(spatial_index.intersection(row.geometry.bounds))
        potential_matches = gdf.iloc[intersecting_indices]
        
        # Filter for polygons with the same attribute value that intersect the current polygon
        potential_matches = potential_matches[
            (potential_matches[attribute] == row[attribute]) & 
            (potential_matches.index != idx) & 
            (potential_matches.geometry.intersects(row.geometry))
        ]

        if not potential_matches.empty:
            # Create a merged polygon from the current polygon and all matching polygons
            polygons_to_merge = [gdf.loc[idx, "geometry"]] + potential_matches.geometry.tolist()

            try:
                # Attempt to merge the polygons
                merged_geometry = unary_union(polygons_to_merge)

                # Update the current polygon's geometry with the merged geometry
                gdf.at[idx, "geometry"] = merged_geometry

                # Mark the matching polygons for removal
                indices_to_drop.update(potential_matches.index)

            except Exception as e:
                # If merge fails, continue with the next polygon
                print(f"Merge failed for index {idx}- {e}")
                continue

    # Remove all merged polygons
    if indices_to_drop:
        gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
        
    # Recalculate area
    gdf = gdf.drop(columns=["area"])
    gdf["area"] = gdf.geometry.area
    
    return gdf



def _remove_child_polygons(gdf, pin):
    """
    Remove 'child polygons' that are contained within 'parent polygons'.
    
    Parameters:
        - gdf: GeoDataFrame containing cadastre polygons
        - pin: Column name representing unique identifiers
    
    Returns:
        - GeoDataFrame with child polygons removed
    """
    # Create a copy to avoid modifying the input while iterating
    result_gdf = gdf.copy()
    
    # Ensure the "child_pin" column exists
    if "child_pin" not in result_gdf.columns:
        result_gdf["child_pin"] = [[] for _ in range(len(result_gdf))]
    
    # Store indices of child polygons to drop
    indices_to_drop = set()
    
    # Iterate over rows
    for index, row in gdf.iterrows():

        # Get all polygons that are within the current row (excluding itself)
        overlapping_rows = result_gdf[result_gdf.geometry.within(row.geometry.buffer(0.1))]
        overlapping_rows = overlapping_rows[overlapping_rows.index != index]

        if not overlapping_rows.empty:
            # Get the area of the parent polygon
            pol_area = row.geometry.area

            # Get the area of all overlapping polygons
            overlapping_rows["child_area"] = overlapping_rows.geometry.area

            # Check if all found polygons are smaller than the current one
            if pol_area > max(overlapping_rows["child_area"]):
                # Store indices of child polygons for removal
                indices_to_drop.update(overlapping_rows.index)

                # Store the pin information of the child polygons
                result_gdf.at[index, "child_pin"] = result_gdf.at[index, "child_pin"] + overlapping_rows[pin].tolist()
    
    # Drop child polygons
    if indices_to_drop:
        result_gdf.drop(index=indices_to_drop, inplace=True)
    
    # Reset index after removal
    result_gdf.reset_index(drop=True, inplace=True)
    
    return result_gdf


def _flatten_twin_polygons(gdf, pin):
    """
    Remove duplicated and overlapping 'twin' polygons with identical areas.

    Parameters:
        - gdf: GeoDataFrame containing cadastre polygons

    Returns:
        - GeoDataFrame with sibling polygons removed
    """
    # Create a copy to avoid modifying the input
    result_gdf = gdf.copy()

    # Create a set to store indices of polygons to drop
    indices_to_drop = set()

    # Iterate over rows
    for index, row in result_gdf.iterrows():
        
        # Skip rows that are already marked for removal
        if index in indices_to_drop:
            continue

        # Find polygons that intersect with the current row and have the same area OR have the exact same geometry
        intersecting_rows = result_gdf[
            ((result_gdf.geometry.intersects(row.geometry)) & 
             (result_gdf["area"] == row["area"])) | 
            (result_gdf["geometry"] == row.geometry)
        ]

        # Exclude the current row itself from intersecting_rows
        intersecting_rows = intersecting_rows[intersecting_rows.index != index]

        # Mark intersecting polygons for removal
        indices_to_drop.update(intersecting_rows.index)

        # Get pin information from the child polygons
        pin_list = intersecting_rows[pin].tolist()

        # Store the pin information of the child polygons in a new column
        result_gdf.at[index, "twin_pin"] = str(pin_list)

    # Drop all marked indices from result_gdf
    result_gdf = result_gdf.drop(index=indices_to_drop).reset_index(drop=True)

    return result_gdf


def _merge_by_filled_polygon(gdf, pin):
    """
    Merge building footprints to the larger polygon using the filled polygon method.
    
    Parameters:
        - gdf: GeoDataFrame containing cadastre polygons
        
    Returns:
        - GeoDataFrame with merged polygons
    """
    # Create spatial index
    spatial_index = _create_spatial_index(gdf)
    
    # Create a set to store indices of polygons to drop
    indices_to_drop = set()
    
    # Create a copy to avoid modifying the input
    result_gdf = gdf.copy()

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
                pin_list = potential_matches[pin].tolist()

                # Store the pin information of the child polygons in a new column
                result_gdf.at[index, "child_pin"] = str(pin_list)

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


def _merge_by_attribute(gdf, attribute):
    """
    Merge intersecting polygons that share the same attribute value.
    
    Parameters:
        - gdf: GeoDataFrame containing cadastre polygons
        - attribute: Column name to use for grouping polygons for merging
        
    Returns:
        - GeoDataFrame with merged polygons
    """
    # Check if the attribute exists in the dataframe
    if attribute not in gdf.columns:
        print(f"Attribute '{attribute}' not found in the GeoDataFrame. Skipping merge.")
        return gdf
    
    # Create spatial index
    spatial_index = create_spatial_index(gdf)
    
    # Create a set to store indices of polygons to drop
    indices_to_drop = set()

    # Iterate over rows
    for idx, row in gdf.iterrows():
        # Skip if this polygon has already been marked for removal
        if idx in indices_to_drop:
            continue

        # Find potential matches using the spatial index
        intersecting_indices = list(spatial_index.intersection(row.geometry.bounds))
        potential_matches = gdf.iloc[intersecting_indices]
        
        # Filter for polygons with the same attribute value that intersect the current polygon
        potential_matches = potential_matches[
            (potential_matches[attribute] == row[attribute]) & 
            (potential_matches.index != idx) & 
            (potential_matches.geometry.intersects(row.geometry))
        ]

        if not potential_matches.empty:
            # Create a merged polygon from the current polygon and all matching polygons
            polygons_to_merge = [gdf.loc[idx, "geometry"]] + potential_matches.geometry.tolist()

            try:
                # Attempt to merge the polygons
                merged_geometry = unary_union(polygons_to_merge)

                # Update the current polygon's geometry with the merged geometry
                gdf.at[idx, "geometry"] = merged_geometry

                # Mark the matching polygons for removal
                indices_to_drop.update(potential_matches.index)

            except Exception as e:
                # If merge fails, continue with the next polygon
                print(f"Merge failed for index {idx}- {e}")
                continue

    # Remove all merged polygons
    if indices_to_drop:
        gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
        
    # Recalculate area
    gdf = gdf.drop(columns=["area"])
    gdf["area"] = gdf.geometry.area
    
    return gdf


def flatten_cadastre(cad_gdf, 
                     pin,
                     proj_crs, 
                     remove_child_polygons=False,
                     flatten_twin_polygons=False,
                     merge_by_filled_polygon=False, 
                     merge_by_convex_hull=False):
    """
    Flatten "parent-child" and "sibling" cadastral polygons. "Parent" polygons contain
    smaller "child" polygons. "Sibling" polygons are perfectly overlapping each other.
    Brisbane cadastres have to be merged by lotplan. Merge building footprints to the parcel polygon.

    Note: Use on any cadastre datasets for any cities.

    Parameters:
        - cad_gdf: Input GeoDataFrame containing cadastre polygons.
        - proj_crs: Projected CRS for calculating area.
        - brisbane: Flag for merging cadastre polygons by lotplan (for Brisbane).
        - adelaide: Flag for dissolving cadastre polygons by plan and volume (for Adelaide).
        - merge_by_filled_polygon: Flag for merging building polygons with parent lot by filled polygon
        - merge_by_convex_hull: Flag for merging building polygons with parent lot by convex hull

    Returns:
        - Flattened GeoDataFrame with retained attributes.
    """
    
    # Make a copy of the input GeoDataFrame
    gdf = cad_gdf.copy()
    
    # Calculate area of each parcel
    gdf = gdf.to_crs(proj_crs)
    gdf["area"] = gdf.geometry.area
    
    if remove_child_polygons:
        # Remove 'child polygons' inside 'parent polygons'
        gdf = _remove_child_polygons(gdf, pin)
    
    if flatten_twin_polygons:
        # Flatten 'sibling polygons'
        gdf = _flatten_twin_polygons(gdf, pin)
    
    # Merge building footprints to the larger polygon using the filled polygon method
    if merge_by_filled_polygon:
        gdf = _merge_by_filled_polygon(gdf, pin)
    
    # Merge building footprints to the larger polygon using the convex hull method
    if merge_by_convex_hull:
        gdf = merge_by_convex_hull(gdf)
    
    return gdf