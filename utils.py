import os
import geopandas as gpd
import pandas as pd
import rtree
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


def dask_within_operation(ddf, aoi):
    """
    Perform a within operation on a Dask GeoDataFrame of cadastral data
    based on an area of interest (AOI) polygon.
    
    Parameters:
        - ddf (dask_geopandas.GeoDataFrame): Dask GeoDataFrame containing targeted polygons.
        - aoi (geopandas.GeoDataFrame: AOI polygon
        
    Returns:
        - GeoDataFrame: GeoDataFrame containing geometries within the AOI polygon.
    """    
    
    # Transform the CRS of the Dask GeoDataFrame
    ddf = ddf.to_crs(aoi.crs)
    
    # Use the within function to find geometries within the AOI polygon
    within_condition = ddf.within(aoi.unary_union)
    
    # Subset the original Dask GeoDataFrame based on the within condition
    ddf_aoi = ddf[within_condition]
    
    # Trigger computation
    ddf_aoi = ddf_aoi.compute()
    
    return ddf_aoi

def dask_intersect_operation(ddf, aoi):
    """
    Perform an intersection operation on a Dask GeoDataFrame of cadastral data
    based on an area of interest (AOI) polygon.
    
    Parameters:
        - ddf (dask_geopandas.GeoDataFrame): Dask GeoDataFrame containing targeted polygons.
        - aoi (geopandas.GeoDataFrame): AOI polygon.
        
    Returns:
        - GeoDataFrame: GeoDataFrame containing geometries that intersect with the AOI polygon.
    """
    
    # Transform the CRS of the Dask GeoDataFrame to match the AOI's CRS
    ddf = ddf.to_crs(aoi.crs)
    
    # Use the intersects function to find geometries that intersect with the AOI polygon
    intersect_condition = ddf.intersects(aoi.unary_union)
    
    # Subset the original Dask GeoDataFrame based on the intersect condition
    ddf_aoi = ddf[intersect_condition]
    
    # Trigger computation
    ddf_aoi = ddf_aoi.compute()
    
    return ddf_aoi


def spatial_join_largest_intersection(proj_crs, out_crs, left_df, right_df, id_col_left, intersection_area_col, id_col_right):
    """
    Spatial join based on largest intersection

    Args:
        proj_crs (str): CRS to be used for projecting geometries
        out_crs (str): CRS of the resulting GeoDataFrame
        left_df (gpd): Variable holding GeoPandas dataframe as left table in spatial join
        right_df (gpd): Variable holding GeoPandas dataframe as right table in spatial join
        id_col_left (str): Column containing unique feature IDs in left table
        intersection_area_col (str): Name of column for storing area of intersection between left and right tables
        id_col_right (str): Column containing unique feature IDs in left table - to append to left table based on largest intersection

    Returns:
        gpd: GeoPandas dataframe of left table with id_col_right appended based on largest intersection
    """    

    # Project both GeoDataFrames to the same coordinate reference system
    left_df = left_df.to_crs(proj_crs)
    right_df = right_df.to_crs(proj_crs)
    
    # Perform spatial overlay to find the intersection between the two GeoDataFrames
    overlay = gpd.overlay(left_df, right_df, how="intersection")
    
    # Calculate the area of intersection for each geometry and add it as a new column
    overlay[intersection_area_col] = overlay.geometry.area
    
    # Keep only the largest intersection for each feature in the left table
    out_df = overlay.sort_values([id_col_left, intersection_area_col], ascending=False).drop_duplicates(subset=[id_col_left], keep="first")
    
    # Select necessary columns for the final output
    out_df = out_df[[id_col_left, intersection_area_col, id_col_right]]
    
    # Merge the selected columns back into the left table
    left_df = left_df.merge(out_df, how="left", on=id_col_left).drop(columns=[intersection_area_col])
    
    # Reproject the resulting GeoDataFrame to the specified output coordinate reference system
    left_df = left_df.to_crs(out_crs)

    return left_df    


def create_spatial_index(gdf):
    idx = rtree.index.Index()
    for i, geometry in enumerate(gdf.geometry):
        idx.insert(i, geometry.bounds)
    return idx       
