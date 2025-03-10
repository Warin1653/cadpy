import os
import geopandas as gpd
import pandas as pd
import rtree
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def create_spatial_index(gdf):
    idx = rtree.index.Index()
    for i, geometry in enumerate(gdf.geometry):
        idx.insert(i, geometry.bounds)
    return idx


def flatten_cadastre(cad_gdf, proj_crs, 
                     brisbane=False, 
                     adelaide=False,
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
        - GeoDataFrame with convex hull geometries (if merge_building_footprints is True).
    """
    
    # Make a copy of the input GeoDataFrame
    gdf = cad_gdf.copy()
    
    # Calculate area of each parcel
    gdf = gdf.to_crs(proj_crs)
    gdf["area"] = gdf.geometry.area
    
    
    # -----------------------------------------------------
    # Flatten 'sibling polygons'
    # -----------------------------------------------------
    
    # Remove polygons with duplicated geometry
    gdf = gdf[~gdf.duplicated(["geometry"], keep="first")]

    # Create a set to store indices of polygons to drop
    indices_to_drop = set()

    # Iterate over rows
    for index, row in gdf.iterrows():
        # Skip rows that are already marked for removal
        if index in indices_to_drop:
            continue

        # Find polygons that intersect with the current row and have the same area
        intersecting_rows = gdf[
            (gdf.geometry.intersects(row.geometry)) & 
            (gdf["area"] == row["area"])
        ]

        # Exclude the current row itself from intersecting_rows
        intersecting_rows = intersecting_rows[intersecting_rows.index != index]

        # Mark intersecting polygons for removal
        indices_to_drop.update(intersecting_rows.index)

    # Drop all marked indices from gdf
    gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
        
    # ---------------------------------------------------
    # Remove 'child polygons' inside 'parent polygons'
    # ---------------------------------------------------

    # Iterate over rows
    for index, row in gdf.iterrows():
    
        # Get all polygons that are within the current row
        overlapping_rows = gdf[gdf.geometry.within(row.geometry.buffer(0.1))]

        # Exclude the current row itself
        overlapping_rows = overlapping_rows[overlapping_rows.index != index]

        # Check if the row has any overlapping polygons
        if not overlapping_rows.empty:

            # Get the area of the polygon being processed
            pol_area = row["area"]

            # Get the area of all overlapping polygons
            overlapping_rows_area = overlapping_rows["area"].tolist()

            # Check if the area of the polygon is larger than all overlapping polygons
            if pol_area > max(overlapping_rows_area):

                # Remove child polygon rows
                gdf.drop(index= overlapping_rows.index, inplace=True)           
      
    # Reset the index of the GeoDataFrame so index of "df" matches with "gdf"
    gdf.reset_index(drop=True, inplace=True)
           
        
    # ---------------------------------------------------
    # Merge polygons by lotplan (for Brisbane)
    # ---------------------------------------------------
    
    # Create spatial index
    spatial_index = create_spatial_index(gdf)
    
    if brisbane:
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
            potential_matches = potential_matches[
                (potential_matches["lotplan"] == row["lotplan"]) & 
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
                    print(f"merge failed for {row.PFI}- {e}")
                    continue

        # Remove all merged polygons
        if indices_to_drop:
            gdf = gdf.drop(index=indices_to_drop)
            
        # Recalculate area
        gdf = gdf.drop(columns=["area"])
        gdf["area"] = gdf.geometry.area
        
        # Rereate the spatial index
        spatial_index = create_spatial_index(gdf)

        
    # ---------------------------------------------------
    # Merge by PLAN (for Adelaide)
    # ---------------------------------------------------
    
    if adelaide:
        if "plan_t" in gdf.columns: 
            # Filter the rows where plan_t is "C" or "S"
            gdf_dissolve = gdf[(gdf["plan_t"] == "C") | (gdf["plan_t"] == "S")]

            # Dissolve by plan
            gdf_dissolve = gdf_dissolve.dissolve(by = "plan")

            # Dissolve by volume
            gdf_dissolve = gdf_dissolve.dissolve(by = "volume")

            # Keep the rest of the parcels (not "C" or "S")
            gdf_remaining = gdf[~(gdf["plan_t"] == "C") | (gdf["plan_t"] == "S")]

            # Concatenate the dissolved and remaining data
            gdf = pd.concat([gdf_remaining, gdf_dissolve]).reset_index(drop=True)
        
        elif "PLAN_T" in gdf.columns:
            # Filter the rows where plan_t is "C" or "S"
            gdf_dissolve = gdf[(gdf["PLAN_T"] == "C") | (gdf["PLAN_T"] == "S")]

            # Dissolve by plan
            gdf_dissolve = gdf_dissolve.dissolve(by = "PLAN")

            # Dissolve by volume
            gdf_dissolve = gdf_dissolve.dissolve(by = "VOLUME")

            # Keep the rest of the parcels (not "C" or "S")
            gdf_remaining = gdf[~(gdf["PLAN_T"] == "C") | (gdf["PLAN_T"] == "S")]

            # Concatenate the dissolved and remaining data
            gdf = pd.concat([gdf_remaining, gdf_dissolve]).reset_index(drop=True)
            
        
        spatial_index = create_spatial_index(gdf)
 

    # ---------------------------------------------------
    # Merge building footprints (within block) to the 
    # larger polygon using convex hull and filled polygon
    # ---------------------------------------------------
    
    if merge_by_filled_polygon:
        
        # ----------------------
        # Filled polygon method
        # ----------------------
        
        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for idx, row in gdf.iterrows():
            # Skip if this polygon has already been marked for removal
            if idx in indices_to_drop:
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
            potential_matches = gdf.iloc[intersecting_indices]
            potential_matches = potential_matches[
                potential_matches.buffer(-1).geometry.within(filled_polygon) & 
                (potential_matches.index != idx)
            ]

            if not potential_matches.empty:
                # Create a merged polygon from the current polygon and all matching polygons
                polygons_to_merge = [row.geometry] + potential_matches.geometry.tolist()

                try:
                    # Attempt to merge the polygons
                    merged_geometry = unary_union(polygons_to_merge)

                    # Update the current polygon's geometry with the merged geometry
                    gdf.at[idx, "geometry"] = merged_geometry

                    # Mark the matching polygons for removal
                    indices_to_drop.update(potential_matches.index)

                except Exception as e:
                    # If merge fails, continue with the next polygon
                    print(f"merge failed for {row.PFI}- {e}")
                    continue

        # Remove all merged polygons
        if indices_to_drop:
            gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
            
            # Rereate the spatial index
            spatial_index = create_spatial_index(gdf)
        
        
    if merge_by_convex_hull:
        # ----------------------
        # Convex hull method
        # ----------------------
        
        # Apply convex hull to each geometry in the GeoDataFrame
        gdf["convex_hull"] = gdf.geometry.apply(lambda geom: geom.convex_hull)

        # Drop original geometry column and convex_hull geometry to actual geometry
        gdf_ch = gdf.drop(columns=["geometry"]).rename(columns={"convex_hull": "geometry"})

        # Calculate convex hull:area ratio
        gdf_ch["ch_area"] = gdf_ch.geometry.area
        gdf_ch["ch_to_area_ratio"] = gdf_ch["ch_area"] / gdf_ch["area"]

        # Create a set to store indices of polygons to drop
        indices_to_drop = set()

        # Iterate over rows
        for idx, row in gdf_ch.iterrows():
            # Skip if this polygon has already been marked for removal
            if idx in indices_to_drop:
                continue

            # Filter out parcels larger than 10,000 sq.m.
            if row["area"] < 5000 and row["ch_to_area_ratio"] < 3:

                # Get the polygons that are within the filled
                intersecting_indices = list(spatial_index.intersection(row.geometry.bounds))
                potential_matches = gdf.iloc[intersecting_indices]
                potential_matches = potential_matches[
                    row.geometry.contains_properly(potential_matches.copy().geometry.buffer(-1)) & 
                    (potential_matches.index != idx)
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
                        print(f"merge failed for {row.PFI}- {e}")
                        continue

        # Remove all merged polygons
        if indices_to_drop:
            gdf = gdf.drop(index=indices_to_drop).reset_index(drop=True)
        
    return gdf



        
