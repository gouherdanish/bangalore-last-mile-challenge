import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import transform
import pyproj
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)
    
class GeoUtils:
    """
    Geographic utilities for coordinate transformations and distance calculations.
    
    This class provides static methods for handling geographic data including
    coordinate reference system (CRS) transformations, UTM zone calculations,
    and distance computations optimized for bus route analysis.
    """
    
    @staticmethod
    def change_crs(shp: gpd.GeoDataFrame, crs: str = 'EPSG:4326') -> gpd.GeoDataFrame:
        """
        Change the coordinate reference system of a GeoDataFrame.
        
        Args:
            shp (gpd.GeoDataFrame): Input GeoDataFrame
            crs (str, optional): Target CRS. Defaults to 'EPSG:4326'.
            
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with updated CRS
        """
        try:
            shp = shp.to_crs(crs)
        except Exception as e:
            logger.warning(f"Failed to transform CRS, setting CRS instead: {e}")
            shp = shp.set_crs(crs)
        finally:
            return shp

    @staticmethod
    def change_crs_of_geom(geom: shapely.geometry.base.BaseGeometry, crs: str = 'EPSG:4326') -> shapely.geometry.base.BaseGeometry:
        """
        Change the coordinate reference system of a single geometry.
        
        Args:
            geom (shapely.geometry.base.BaseGeometry): Input geometry
            crs (str, optional): Target CRS. Defaults to 'EPSG:4326'.
            
        Returns:
            shapely.geometry.base.BaseGeometry: Geometry with updated CRS
        """
        shp = gpd.GeoDataFrame(geometry=[geom], crs=crs)
        shp = GeoUtils.change_crs(shp)
        return shp.geometry[0]
        
    @staticmethod
    def collect_utm_crs(utm_zone: int, is_south: bool) -> str:
        """
        Create UTM CRS string from zone and hemisphere information.
        
        Args:
            utm_zone (int): UTM zone number
            is_south (bool): Whether the location is in southern hemisphere
            
        Returns:
            str: UTM CRS string
        """
        crs = pyproj.CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'south': is_south})
        return ':'.join(crs.to_authority())

    @staticmethod
    def get_utm_zone(longitude: float) -> int:
        """
        Calculate UTM zone from longitude.
        
        Args:
            longitude (float): Longitude in decimal degrees
            
        Returns:
            int: UTM zone number
        """
        return int(31 + longitude//6)
    
    @staticmethod
    def is_southern_hemisphere(latitude: float) -> bool:
        """
        Check if latitude is in southern hemisphere.
        
        Args:
            latitude (float): Latitude in decimal degrees
            
        Returns:
            bool: True if in southern hemisphere, False otherwise
        """
        return True if latitude < 0 else False

    @staticmethod
    def get_proj_crs(lon: float, lat: float) -> str:
        """
        Get appropriate UTM CRS for given coordinates.
        
        Args:
            lon (float): Longitude in decimal degrees
            lat (float): Latitude in decimal degrees
            
        Returns:
            str: UTM CRS string
        """
        is_south = GeoUtils.is_southern_hemisphere(lat)
        utm_zone = GeoUtils.get_utm_zone(lon)
        return GeoUtils.collect_utm_crs(utm_zone, is_south)

    @staticmethod
    def calculate_dist(p1: shapely.geometry.Point, p2: shapely.geometry.Point) -> float:
        """
        Calculate distance between two points in meters.
        
        Args:
            p1 (shapely.geometry.Point): First point
            p2 (shapely.geometry.Point): Second point
            
        Returns:
            float: Distance in meters
        """
        local_crs = GeoUtils.get_proj_crs(lon=p1.x, lat=p1.y)
        p1_proj = gpd.GeoDataFrame(geometry=[p1], crs=4326).to_crs(local_crs).geometry[0]
        p2_proj = gpd.GeoDataFrame(geometry=[p2], crs=4326).to_crs(local_crs).geometry[0]
        return p1_proj.distance(p2_proj)

    @staticmethod
    def calculate_dist_vectorized(points1: gpd.GeoSeries, points2: gpd.GeoSeries) -> np.ndarray:
        """
        Vectorized distance calculation between two sets of points.
        
        This method efficiently calculates distances between corresponding points
        in two GeoSeries using appropriate UTM projection for accuracy.
        
        Args:
            points1 (gpd.GeoSeries): First set of points
            points2 (gpd.GeoSeries): Second set of points
            
        Returns:
            np.ndarray: Array of distances in meters
        """
        # Get representative point for CRS
        ref_point = points1.iloc[0]
        local_crs = GeoUtils.get_proj_crs(lon=ref_point.x, lat=ref_point.y)
        
        # Project both sets of points at once
        points1_proj = gpd.GeoSeries(points1, crs=4326).to_crs(local_crs)
        points2_proj = gpd.GeoSeries(points2, crs=4326).to_crs(local_crs)
        
        # Calculate distances vectorized
        return points1_proj.distance(points2_proj).values

