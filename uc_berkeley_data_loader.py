#!/usr/bin/env python3
"""
UC Berkeley Smart Building System - Real Dataset Loader
Processes the actual UC Berkeley dataset to match the simplified schema structure
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import glob
from pathlib import Path

def load_real_uc_berkeley_data(
    dataset_path: str = "f:/UCSD/Project/uc_berkeley_dataset/KETI",
    id_mapping_csv: str = "f:/UCSD/Project/id_mapping.csv",
    hours_limit: int = None,
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    Load real UC Berkeley Smart Building System dataset and convert to match graph database schema
    
    Args:
        dataset_path: Path to the extracted KETI dataset folder
        id_mapping_csv: Path to the id_mapping.csv file that defines room-floor-id mapping
        hours_limit: Limit data to recent N hours (None for at least 1 week of data)
        output_csv_path: Path to save processed data as CSV (optional)
    Returns:
        DataFrame with columns: timestamp, sensor_reading, sensor_id
    """
    
    print(f"Loading Real UC Berkeley Smart Building System Dataset...")
    print(f"   Dataset path: {dataset_path}")
    print(f"   ID mapping CSV: {id_mapping_csv}")
    
    # Load the ID mapping CSV to match graph database structure
    try:
        id_mapping_df = pd.read_csv(id_mapping_csv)
        print(f"   Loaded ID mapping with {len(id_mapping_df)} rooms")
        print(f"   ID mapping preview:")
        print(id_mapping_df.to_string(index=False))
    except Exception as e:
        raise ValueError(f"Failed to load ID mapping CSV: {e}")
      # Create room mapping from CSV
    room_mapping = {}
    for _, row in id_mapping_df.iterrows():
        room_num = str(row['room'])  # Original room number
        room_mapping[room_num] = {
            'floor': int(row['floor']),
            'room_number': int(row['room']),
            'base_id': int(row['id'])
        }
    
    print(f"   Room mapping from CSV: {room_mapping}")
    
    # Verify which rooms from id_mapping.csv actually exist in the dataset
    available_rooms = []
    missing_rooms = []
    
    for room_num in room_mapping.keys():
        room_path = os.path.join(dataset_path, room_num)
        if os.path.exists(room_path):
            available_rooms.append(room_num)
        else:            missing_rooms.append(room_num)
    
    print(f"   Rooms from id_mapping.csv found in dataset: {sorted(available_rooms)}")
    if missing_rooms:
        print(f"   Rooms from id_mapping.csv NOT found in dataset: {sorted(missing_rooms)}")
    
    # Show what sensor IDs will be generated for each room (to match graph database)
    print(f"\n🔗 Sensor ID Generation Preview (Graph Database Compatible):")
    sensor_types_ordered = ['tempreture', 'light', 'co2', 'motion']  # Match graph DB order
    for room_num in sorted(available_rooms):
        room_info = room_mapping[room_num]
        base_id = room_info['base_id']
        floor = room_info['floor']
        print(f"   Room {room_num} (Floor {floor}, Base ID {base_id}):")
        for i, sensor_type in enumerate(sensor_types_ordered):
            sensor_id = f"{sensor_type[0].upper()}{base_id }"
            print(f"     {sensor_type}: {sensor_id}")
    
    # Define sensor type mappings to match graph database schema
    # Graph DB uses: ["tempreture", "light", "co2", "motion"] (note: misspelled temperature)
    # Dataset has: co2, humidity, light, pir, temperature
    sensor_mapping = {
        'temperature': 'tempreture',  # Match graph DB spelling
        'co2': 'co2',
        'light': 'light',
        'pir': 'motion'  # Map PIR (motion) to motion
        # Note: skipping humidity as it's not in the graph schema
    }
    all_data = []
    
    # Process ONLY the rooms from the ID mapping that exist in the dataset
    for original_room, room_info in room_mapping.items():
        room_path = os.path.join(dataset_path, original_room)
        
        # Check if room directory exists in dataset
        if not os.path.exists(room_path):
            print(f"   Room directory {original_room} not found in dataset, skipping...")
            continue
        
        print(f"   Processing room {original_room} -> Floor {room_info['floor']}, Base ID: {room_info['base_id']}")
        
        # Process each sensor type to match graph database schema
        sensor_index = 0
        for sensor_file, sensor_type in sensor_mapping.items():
            csv_path = os.path.join(room_path, f"{sensor_file}.csv")
            
            if not os.path.exists(csv_path):
                print(f"     Missing {sensor_file}.csv in {original_room}")
                sensor_index += 1  # Still increment to maintain ID consistency
                continue
                
            try:
                # Read the CSV file
                df = pd.read_csv(csv_path, header=None, names=['timestamp', 'value'])
                # Convert Unix timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                # Get at least 1 week of data (or all data if less than 1 week)
                if hours_limit:
                    cutoff_time = df['timestamp'].max() - timedelta(hours=hours_limit)
                    df = df[df['timestamp'] >= cutoff_time]
                else:
                    # Default: extract at least 1 week of data at original interval
                    total_duration = df['timestamp'].max() - df['timestamp'].min()
                    if total_duration >= timedelta(days=7):
                        # Take the most recent week
                        cutoff_time = df['timestamp'].max() - timedelta(days=7)
                        df = df[df['timestamp'] >= cutoff_time]
                    # If less than a week available, use all data
                
                # Process sensor readings
                df['sensor_reading'] = df['value'].astype(float)
                  # Generate sensor_id using graph database format: [first_letter][base_id * 4 + sensor_index + 1]
                base_id = room_info['base_id']
                sensor_id = f"{sensor_type[0].upper()}{base_id}"
                df['sensor_id'] = sensor_id
                df['sensor_type'] = sensor_type
                
                # Select only the columns we need for new 3-column schema
                processed_df = df[['timestamp', 'sensor_reading', 'sensor_id', 'sensor_type']].copy()
                
                all_data.append(processed_df)
                
                print(f"      {sensor_type} (ID: {sensor_id}): {len(processed_df)} records")
                
            except Exception as e:
                print(f"      Error processing {sensor_file}.csv: {e}")
            finally:
                sensor_index += 1  # Always increment to maintain consistent ID sequence
    
    # Combine all data
    if not all_data:
        raise ValueError("No data was successfully loaded from the dataset")
    
    final_df = pd.concat(all_data, ignore_index=True)    # Sort by timestamp
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    
    # Save as CSV if path provided
    if output_csv_path:
        final_df.to_csv(output_csv_path, index=False)
        print(f"Saved processed data to: {output_csv_path}")
    print(f"\nDataset Loading Summary:")
    print(f"   Total records: {len(final_df):,}")
    print(f"   Unique sensors: {final_df['sensor_id'].nunique()}")
    print(f"   Time range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
    print(f"   Duration: {final_df['timestamp'].max() - final_df['timestamp'].min()}")
      # Show sensor distribution
    print(f"\nSensor Distribution:")
    sensor_counts = final_df['sensor_id'].value_counts()
    for sensor_id, count in sensor_counts.head(15).items():  # Show more sensors
        print(f"   {sensor_id}: {count:,} records")
    
    # Show sample data
    print(f"\nSample Data (Graph Database Compatible Schema):")
    print(final_df.head(10).to_string(index=False))
    
    # Show sensor ID format verification
    print(f"\nSensor ID Format Verification:")
    unique_sensors = final_df['sensor_id'].unique()
    print(f"   Sample sensor IDs: {sorted(unique_sensors)}")
    
    # Validate schema compliance
    expected_columns = {'timestamp', 'sensor_reading', 'sensor_id', 'sensor_type'}
    actual_columns = set(final_df.columns)
    
    if expected_columns == actual_columns:
        print(f"\n Schema validation passed! Compatible with graph database.")
    else:
        print(f"\nSchema validation failed!")
        print(f"   Expected: {expected_columns}")
        print(f"   Actual: {actual_columns}")
    
    return final_df


def get_building_structure_from_real_data(id_mapping_csv: str = "f:/UCSD/Project/id_mapping.csv") -> Dict[str, Any]:
    """
    Extract building structure information from the ID mapping CSV for Neo4j graph creation
    
    Args:
        id_mapping_csv: Path to the id_mapping.csv file
        
    Returns:
        Dictionary with building structure information compatible with graph database
    """
    
    try:
        id_mapping_df = pd.read_csv(id_mapping_csv)
    except Exception as e:
        raise ValueError(f"Failed to load ID mapping CSV: {e}")
    
    building_structure = {
        'floors': {},
        'sensor_types': ['tempreture', 'light', 'co2', 'motion'],  # Match graph DB schema
        'room_mapping': {}
    }
    
    # Create structure based on CSV data
    for _, row in id_mapping_df.iterrows():
        room_num = int(row['room'])
        floor_num = int(row['floor'])
        base_id = int(row['id'])
        
        if floor_num not in building_structure['floors']:
            building_structure['floors'][floor_num] = []
        
        room_info = {
            'room_number': room_num,
            'base_id': base_id
        }
        
        building_structure['floors'][floor_num].append(room_info)
        building_structure['room_mapping'][room_num] = {
            'floor': floor_num,
            'base_id': base_id
        }
    
    return building_structure


def validate_real_dataset_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the loaded real dataset matches our expected graph database compatible schema
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    
    print("🔍 Validating Real Dataset Schema for Graph Database Compatibility...")
    
    # Check columns for graph database compatible schema
    expected_columns = {'sensor_id', 'sensor_reading', 'sensor_type', 'timestamp'}
    actual_columns = set(df.columns)
    
    if expected_columns != actual_columns:
        print(f"Column mismatch: Expected {expected_columns}, got {actual_columns}")
        return False
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        print(f" Timestamp column is not datetime: {df['timestamp'].dtype}")
        return False
    
    if not pd.api.types.is_numeric_dtype(df['sensor_reading']):
        print(f"Sensor reading column is not numeric: {df['sensor_reading'].dtype}")
        return False
    
    # Check sensor_id format (should match graph database format: [letter][number])
    sample_ids = df['sensor_id'].head(10).tolist()
    for sensor_id in sample_ids:
        if not (len(sensor_id) >= 2 and sensor_id[0].isalpha() and sensor_id[1:].isdigit()):
            print(f"Invalid sensor_id format: {sensor_id} (expected format: [letter][number], e.g., t5, l6)")
            return False
    
    # Check sensor types match graph database schema
    valid_sensor_types = {'tempreture', 'light', 'co2', 'motion'}
    actual_sensor_types = set(df['sensor_type'].unique())
    if not actual_sensor_types.issubset(valid_sensor_types):
        print(f"Invalid sensor types: {actual_sensor_types - valid_sensor_types}")
        print(f"   Valid types: {valid_sensor_types}")
        return False
    
    # Check for reasonable data ranges by sensor type
    sensor_stats = df.groupby('sensor_type')['sensor_reading'].agg(['min', 'max', 'mean'])
    print(" Sensor Reading Ranges by Type:")
    for sensor_type, stats in sensor_stats.iterrows():
        print(f"   {sensor_type}: {stats['min']:.1f} to {stats['max']:.1f} (avg: {stats['mean']:.1f})")
    
    print("Schema validation passed! Data is compatible with graph database schema.")
    return True


if __name__ == "__main__":
    """Test the real data loader with graph database compatible schema"""
    
    print("Testing Real UC Berkeley Dataset Loader - Graph Database Compatible")
    print("="*70)
    
    try:
        # Load the real dataset using id_mapping.csv
        print("Loading real UC Berkeley dataset using id_mapping.csv structure...")
        output_csv = "f:/UCSD/Project/uc_berkeley_processed_data_graph_compatible.csv"
        real_data = load_real_uc_berkeley_data(
            hours_limit=None,  # Get at least 1 week
            output_csv_path=output_csv
        )
        
        # Validate schema
        is_valid = validate_real_dataset_schema(real_data)
        
        # Get building structure
        building_structure = get_building_structure_from_real_data()
        
        print(f"\n Building Structure (from id_mapping.csv):")
        for floor, rooms in building_structure['floors'].items():
            room_numbers = [r['room_number'] for r in rooms]
            base_ids = [r['base_id'] for r in rooms]
            print(f"Floor {floor}: Rooms {room_numbers} (Base IDs: {base_ids})")
        
        print(f"\nSensor Types: {building_structure['sensor_types']}")
        
        print(f"\nGraph database compatible dataset loading completed successfully!")
        print(f"Ready for Neo4j graph database integration")
        print(f"CSV saved to: {output_csv}")
        
        # Show sample sensor IDs that match graph database format
        print(f"\n🆔 Sample Sensor IDs (Graph Database Compatible):")
        sample_sensors = real_data['sensor_id'].unique()[:10]
        for sensor_id in sample_sensors:
            sensor_type = real_data[real_data['sensor_id'] == sensor_id]['sensor_type'].iloc[0]
            print(f"   {sensor_id} ({sensor_type})")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
