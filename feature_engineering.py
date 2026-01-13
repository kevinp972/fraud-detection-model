"""
Feature Engineering Module for Fraud Detection

This module intelligently constructs only the requested features by parsing
feature names and executing minimal required operations.

Key Principle:
    - df_historical: Historical data used to compute aggregations (train set)
    - df_current: Current transactions to generate features for (test set)
    - selected_features: List of specific feature names to construct
    
Usage:
    from feature_engineering import construct_features
    
    features_df = construct_features(
        df_historical=df_train,
        df_current=df_test,
        selected_features=['Cardnum_vdratio_1by14', 'card_dow_actual/avg_7', ...],
        verbose=True
    )
"""

import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
from geopy.distance import geodesic
from sklearn.preprocessing import TargetEncoder
import gc
import re
from collections import defaultdict


def parse_feature_name(feature_name):
    """
    Returns:
        Dictionary with parsed components or None if not recognized
    """
    # Velocity-to-recency ratio: entity_vdratio_ShortbyLong
    match = re.match(r'(.+)_vdratio_(\d+)by(\d+)$', feature_name)
    if match:
        return {
            'type': 'vdratio',
            'entity': match.group(1),
            'short': int(match.group(2)),
            'long': int(match.group(3))
        }
    
    # Cross-entity unique count: entityI_unique_count_for_entityJ_window
    match = re.match(r'(.+)_unique_count_for_(.+)_(\d+)$', feature_name)
    if match:
        return {
            'type': 'unique_count',
            'entity_i': match.group(1),
            'entity_j': match.group(2),
            'window': int(match.group(3))
        }
    
    # Actual/stat ratio: entity_actual/stat_window
    match = re.match(r'(.+)_actual/(avg|max|med|total)_(\d+)$', feature_name)
    if match:
        return {
            'type': f'actual_{match.group(2)}',
            'entity': match.group(1),
            'stat': match.group(2),
            'window': int(match.group(3))
        }
    
    # Variability: entity_variability_stat_window
    match = re.match(r'(.+)_variability_(avg|max|med)_(\d+)$', feature_name)
    if match:
        return {
            'type': f'variability_{match.group(2)}',
            'entity': match.group(1),
            'stat': match.group(2),
            'window': int(match.group(3))
        }
    
    # Count velocity squared: entity_count_short_by_long_sq
    match = re.match(r'(.+)_count_(\d+)_by_(\d+)_sq$', feature_name)
    if match:
        return {
            'type': 'count_velocity_sq',
            'entity': match.group(1),
            'short': int(match.group(2)),
            'long': int(match.group(3))
        }
    
    # Count velocity: entity_count_short_by_long
    match = re.match(r'(.+)_count_(\d+)_by_(\d+)$', feature_name)
    if match:
        return {
            'type': 'count_velocity',
            'entity': match.group(1),
            'short': int(match.group(2)),
            'long': int(match.group(3))
        }
    
    # Total amount velocity: entity_total_amount_short_by_long
    match = re.match(r'(.+)_total_amount_(\d+)_by_(\d+)$', feature_name)
    if match:
        return {
            'type': 'total_velocity',
            'entity': match.group(1),
            'short': int(match.group(2)),
            'long': int(match.group(3))
        }
    
    # Time-based aggregations: entity_stat_window
    match = re.match(r'(.+)_(count|avg|max|med|total)_(\d+)$', feature_name)
    if match:
        return {
            'type': f'time_{match.group(2)}',
            'entity': match.group(1),
            'stat': match.group(2),
            'window': int(match.group(3))
        }
    
    # Day since last (no window)
    match = re.match(r'(.+)_day_since$', feature_name)
    if match:
        return {
            'type': 'day_since',
            'entity': match.group(1)
        }
    
    # Target encoding: column_TE
    if feature_name.endswith('_TE'):
        return {
            'type': 'target_encoding',
            'column': feature_name[:-3]
        }
    
    # Business logic features (hardcoded patterns)
    business_features = [
        'distance_to_last_transaction', 'suspicious_distance_flag',
        'amount_category', 'foreign_zip_flag', 'is_weekend',
        'amount_is_rounded', 'high_amount_for_merchant',
        'merchant_dominance_score', 'state_inconsistency_flag'
    ]
    if feature_name in business_features:
        return {
            'type': 'business_logic',
            'name': feature_name
        }
    
    return None


def analyze_required_features(selected_features):
    """
    Analyze selected features to determine what base features need to be computed.
    
    Returns:
        Dictionary with entities, windows, base_features, derived_features, etc.
    """
    entities = set()
    time_windows = set()
    base_features = {}
    derived_features = {}
    te_columns = set()
    business_features = set()
    
    for feat in selected_features:
        parsed = parse_feature_name(feat)
        
        if parsed is None:
            print(f"Warning: Could not parse feature '{feat}'")
            continue
        
        feat_type = parsed['type']
        
        # Target encoding
        if feat_type == 'target_encoding':
            te_columns.add(parsed['column'])
            continue
        
        # Business logic
        if feat_type == 'business_logic':
            business_features.add(parsed['name'])
            continue
        
        # Features that need entity
        if 'entity' in parsed:
            entities.add(parsed['entity'])
        
        # Velocity-derived features (need base count/total features)
        if feat_type == 'vdratio':
            entities.add(parsed['entity'])
            time_windows.add(parsed['short'])
            time_windows.add(parsed['long'])
            # Need count features and day_since
            base_features[f"{parsed['entity']}_count_{parsed['short']}"] = {
                'type': 'time_count', 'entity': parsed['entity'], 'window': parsed['short']
            }
            base_features[f"{parsed['entity']}_count_{parsed['long']}"] = {
                'type': 'time_count', 'entity': parsed['entity'], 'window': parsed['long']
            }
            base_features[f"{parsed['entity']}_day_since"] = {
                'type': 'day_since', 'entity': parsed['entity']
            }
            derived_features[feat] = parsed
        
        elif feat_type == 'count_velocity' or feat_type == 'count_velocity_sq':
            entities.add(parsed['entity'])
            time_windows.add(parsed['short'])
            time_windows.add(parsed['long'])
            # Need count features
            base_features[f"{parsed['entity']}_count_{parsed['short']}"] = {
                'type': 'time_count', 'entity': parsed['entity'], 'window': parsed['short']
            }
            base_features[f"{parsed['entity']}_count_{parsed['long']}"] = {
                'type': 'time_count', 'entity': parsed['entity'], 'window': parsed['long']
            }
            derived_features[feat] = parsed
        
        elif feat_type == 'total_velocity':
            entities.add(parsed['entity'])
            time_windows.add(parsed['short'])
            time_windows.add(parsed['long'])
            # Need total features
            base_features[f"{parsed['entity']}_total_{parsed['short']}"] = {
                'type': 'time_total', 'entity': parsed['entity'], 'window': parsed['short']
            }
            base_features[f"{parsed['entity']}_total_{parsed['long']}"] = {
                'type': 'time_total', 'entity': parsed['entity'], 'window': parsed['long']
            }
            derived_features[feat] = parsed
        
        elif feat_type.startswith('actual_'):
            entities.add(parsed['entity'])
            time_windows.add(parsed['window'])
            # Need the base stat feature
            stat = parsed['stat']
            base_features[f"{parsed['entity']}_{stat}_{parsed['window']}"] = {
                'type': f'time_{stat}', 'entity': parsed['entity'], 'window': parsed['window']
            }
            derived_features[feat] = parsed
        
        elif feat_type.startswith('variability_'):
            entities.add(parsed['entity'])
            time_windows.add(parsed['window'])
            derived_features[feat] = parsed
        
        elif feat_type == 'unique_count':
            entities.add(parsed['entity_i'])
            entities.add(parsed['entity_j'])
            time_windows.add(parsed['window'])
            derived_features[feat] = parsed
        
        # Base time features
        elif feat_type.startswith('time_') or feat_type == 'day_since':
            entities.add(parsed['entity'])
            if 'window' in parsed:
                time_windows.add(parsed['window'])
            base_features[feat] = parsed
    
    return {
        'entities': entities,
        'time_windows': sorted(time_windows),
        'base_features': base_features,
        'derived_features': derived_features,
        'te_columns': te_columns,
        'business_features': business_features
    }


def compute_time_based_features_efficient(df_historical, df_current, entity, windows):
    """
    Compute time-based features for a single entity across multiple windows efficiently.
    
    Returns:
        Dictionary of features {feature_name: Series}
    """
    features = {}
    
    df_main = df_current.copy()
    df_past = df_historical.copy()
    
    df_past['past_date'] = df_past['Date']
    df_past['past_recnum'] = df_past['Recnum']
    
    # Self-join
    merged = pd.merge(
        df_main[['Recnum', 'Date', entity]],
        df_past[['past_recnum', 'past_date', entity, 'Amount']],
        on=entity
    )
    
    # Filter to only past transactions
    past_txns = merged[merged['Recnum'] > merged['past_recnum']]
    
    # DAYS SINCE LAST TRANSACTION
    last_txn = past_txns.groupby('Recnum')[['Date', 'past_date']].last()
    days_since = (last_txn['Date'] - last_txn['past_date']).dt.days
    earliest_date = df_historical['Date'].min()
    features[f'{entity}_day_since'] = df_main['Recnum'].map(days_since).fillna(
        (df_main['Date'] - earliest_date).dt.days
    )
    
    # ROLLING TIME WINDOW FEATURES
    for days in windows:
        window_txns = past_txns[
            past_txns['past_date'] >= (past_txns['Date'] - dt.timedelta(days=days))
        ][['Recnum', entity, 'Amount']]
        
        # Count
        txn_counts = window_txns.groupby('Recnum')[entity].count()
        features[f'{entity}_count_{days}'] = df_main['Recnum'].map(txn_counts)
        
        # Amount aggregations
        amount_agg = window_txns.groupby('Recnum')['Amount'].agg(['mean', 'max', 'median', 'sum'])
        features[f'{entity}_avg_{days}'] = df_main['Recnum'].map(amount_agg['mean'])
        features[f'{entity}_max_{days}'] = df_main['Recnum'].map(amount_agg['max'])
        features[f'{entity}_med_{days}'] = df_main['Recnum'].map(amount_agg['median'])
        features[f'{entity}_total_{days}'] = df_main['Recnum'].map(amount_agg['sum'])
    
    del merged, past_txns
    gc.collect()
    
    return features


def compute_variability_features_efficient(df_historical, df_current, entity, windows):
    """
    Compute amount variability features for a single entity.
    
    Returns:
        Dictionary of features {feature_name: Series}
    """
    features = {}
    
    df_main = df_current.copy()
    df_past = df_historical.copy()
    df_past['past_recnum'] = df_past['Recnum']
    df_past['past_date'] = df_past['Date']
    
    merged = pd.merge(
        df_main[['Recnum', 'Date', entity, 'Amount']],
        df_past[['past_recnum', 'past_date', entity, 'Amount']],
        on=entity,
        suffixes=('_current', '_past')
    )
    
    past_txns = merged[merged['Recnum'] > merged['past_recnum']]
    
    for days in windows:
        window_txns = past_txns[
            past_txns['past_date'] >= (past_txns['Date'] - dt.timedelta(days=days))
        ].copy()
        
        window_txns['amount_diff'] = window_txns['Amount_past'] - window_txns['Amount_current']
        
        variability_agg = window_txns.groupby('Recnum')['amount_diff'].agg(['mean', 'max', 'median'])
        features[f'{entity}_variability_avg_{days}'] = df_main['Recnum'].map(variability_agg['mean'])
        features[f'{entity}_variability_max_{days}'] = df_main['Recnum'].map(variability_agg['max'])
        features[f'{entity}_variability_med_{days}'] = df_main['Recnum'].map(variability_agg['median'])
    
    del merged, past_txns
    gc.collect()
    
    return features


def compute_unique_count_efficient(df_historical, df_current, entity_i, entity_j, windows):
    """
    Compute unique count of entity_j within entity_i's time windows.
    
    Returns:
        Dictionary of features {feature_name: Series}
    """
    features = {}
    
    df_main = df_current.copy()
    df_past = df_historical.copy()
    df_past['past_recnum'] = df_past['Recnum']
    df_past['past_date'] = df_past['Date']
    
    merged = pd.merge(
        df_main[['Recnum', 'Date', entity_i]],
        df_past[['past_recnum', 'past_date', entity_i, entity_j]],
        on=entity_i
    )
    
    past_txns = merged[merged['Recnum'] > merged['past_recnum']]
    
    for days in windows:
        window_txns = past_txns[
            past_txns['past_date'] >= (past_txns['Date'] - dt.timedelta(days=days))
        ]
        
        unique_counts = window_txns.groupby('Recnum')[entity_j].nunique()
        features[f'{entity_i}_unique_count_for_{entity_j}_{days}'] = df_main['Recnum'].map(unique_counts)
    
    del merged, past_txns
    gc.collect()
    
    return features


def compute_business_logic_features(df_historical, df_current, business_features,
                                    zip_coords=None, us_zip_set=None):
    """
    Compute requested business logic features.
    
    Returns:
        Dictionary of features
    """
    features = {}
    
    for feat_name in business_features:
        
        if feat_name == 'distance_to_last_transaction' and zip_coords is not None:
            df_temp = df_current.copy()
            df_temp['prev_merch_zip'] = df_temp.groupby('Cardnum')['Merch zip'].shift(1)
            
            def calculate_distance(zip1, zip2):
                coords_1 = zip_coords.get(str(zip1))
                coords_2 = zip_coords.get(str(zip2))
                if coords_1 and coords_2:
                    return geodesic(
                        (coords_1['latitude'], coords_1['longitude']), 
                        (coords_2['latitude'], coords_2['longitude'])
                    ).miles
                return None
            
            features[feat_name] = df_temp.apply(
                lambda row: calculate_distance(row['prev_merch_zip'], row['Merch zip']), 
                axis=1
            )
        
        elif feat_name == 'suspicious_distance_flag':
            if 'distance_to_last_transaction' in features:
                features[feat_name] = (features['distance_to_last_transaction'] > 1000).astype(int)
        
        elif feat_name == 'amount_category':
            amount_bins = pd.qcut(df_historical['Amount'], q=5, duplicates='drop', retbins=True)[1]
            features[feat_name] = pd.cut(
                df_current['Amount'], 
                bins=amount_bins, 
                labels=range(1, len(amount_bins)),
                include_lowest=True
            ).astype(float)
        
        elif feat_name == 'foreign_zip_flag' and us_zip_set is not None:
            features[feat_name] = df_current['Merch zip'].apply(
                lambda x: 0 if str(x) in us_zip_set else 1
            )
        
        elif feat_name == 'is_weekend':
            if 'DayOfWeek' in df_current.columns:
                features[feat_name] = (df_current['DayOfWeek'] >= 5).astype(int)
        
        elif feat_name == 'amount_is_rounded':
            features[feat_name] = (df_current['Amount'] % 1 == 0).astype(int)
        
        elif feat_name == 'high_amount_for_merchant':
            merchant_avg = df_historical.groupby('Merchnum')['Amount'].mean()
            df_current_merchant_avg = df_current['Merchnum'].map(merchant_avg)
            features[feat_name] = (df_current['Amount'] > (2 * df_current_merchant_avg)).astype(int)
        
        elif feat_name == 'merchant_dominance_score':
            card_merchant_counts = df_historical.groupby(['Cardnum', 'Merchnum']).size().reset_index(name='txn_count')
            most_freq_merchant = card_merchant_counts.loc[
                card_merchant_counts.groupby('Cardnum')['txn_count'].idxmax()
            ]
            total_txns_per_card = df_historical.groupby('Cardnum').size().reset_index(name='total_txns')
            dominance = pd.merge(most_freq_merchant, total_txns_per_card, on='Cardnum')
            dominance['merchant_dominance_score'] = dominance['txn_count'] / dominance['total_txns']
            features[feat_name] = df_current['Cardnum'].map(
                dominance.set_index('Cardnum')['merchant_dominance_score']
            )
        
        elif feat_name == 'state_inconsistency_flag':
            df_sorted = df_current.sort_values(['Cardnum', 'Date']).copy()
            df_sorted['prev_state'] = df_sorted.groupby('Cardnum')['Merch state'].shift(1)
            df_sorted['prev_date'] = df_sorted.groupby('Cardnum')['Date'].shift(1)
            df_sorted['days_since_prev'] = (df_sorted['Date'] - df_sorted['prev_date']).dt.days
            state_inconsistency = (
                (df_sorted['Merch state'] != df_sorted['prev_state']) & 
                (df_sorted['days_since_prev'] <= 1)
            ).fillna(False).astype(int)
            features[feat_name] = df_current.index.map(dict(zip(df_sorted.index, state_inconsistency)))
    
    return features


# Main feature construction function

def construct_features(df_historical, df_current, selected_features, 
                      zip_coords=None, us_zip_set=None, verbose=True):
    """
    Intelligently construct only the requested features by parsing feature names
    and executing minimal required operations.
    
    Args:
        df_historical: Historical data used to compute aggregations (training set)
        df_current: Current transactions to generate features for (test set)
        selected_features: List of specific feature names to construct
        zip_coords: Dictionary mapping ZIP codes to coordinates (for geographic features)
        us_zip_set: Set of valid US ZIP codes (for business logic features)
        verbose: Whether to show progress
    
    Returns:
        DataFrame with only the selected features (indexed by df_current)
    """
    if verbose:
        print(f"Historical data: {df_historical.shape[0]} rows")
        print(f"Current data: {df_current.shape[0]} rows")
        print(f"Selected features: {len(selected_features)}")
    
    # Ensure Date is datetime
    for df in [df_historical, df_current]:
        if 'Date' not in df.columns:
            raise ValueError("'Date' column is required")
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
    
    # Analyze what features are needed
    if verbose:
        print("\n[1/3] Analyzing feature requirements...")
    
    requirements = analyze_required_features(selected_features)
    
    if verbose:
        print(f"  Entities needed: {len(requirements['entities'])}")
        print(f"  Time windows needed: {requirements['time_windows']}")
        print(f"  Base features: {len(requirements['base_features'])}")
        print(f"  Derived features: {len(requirements['derived_features'])}")
        print(f"  Target encoding: {len(requirements['te_columns'])}")
        print(f"  Business features: {len(requirements['business_features'])}")
    
    all_features = {}
    
    # Step 1: Target Encoding
    if requirements['te_columns']:
        if verbose:
            print("\n[2/3] Computing target encoding...")
        
        for col in requirements['te_columns']:
            if col not in df_historical.columns or col not in df_current.columns:
                if verbose:
                    print(f"  Warning: {col} not found, skipping")
                continue
            
            encoder = TargetEncoder(smooth='auto', target_type='binary', random_state=42)
            encoder.fit(df_historical[[col]], df_historical['Fraud'])
            all_features[f'{col}_TE'] = encoder.transform(df_current[[col]]).ravel()
    
    # Step 2: Compute base features efficiently
    if requirements['base_features'] or requirements['derived_features']:
        if verbose:
            print("\n[3/3] Computing features...")
        
        # Group by entity and windows to minimize operations
        entity_to_windows = defaultdict(set)
        entity_to_variability_windows = defaultdict(set)
        entity_pairs_to_windows = defaultdict(set)
        
        # Collect what we need for each entity
        for feat_name, parsed in requirements['base_features'].items():
            if parsed['type'].startswith('time_'):
                if 'window' in parsed:
                    entity_to_windows[parsed['entity']].add(parsed['window'])
        
        for feat_name, parsed in requirements['derived_features'].items():
            if parsed['type'].startswith('variability_'):
                entity_to_variability_windows[parsed['entity']].add(parsed['window'])
            elif parsed['type'] == 'unique_count':
                key = (parsed['entity_i'], parsed['entity_j'])
                entity_pairs_to_windows[key].add(parsed['window'])
            elif parsed['type'] in ['vdratio', 'count_velocity', 'count_velocity_sq', 'total_velocity']:
                # These need count/total features which are already in base_features
                pass
            elif parsed['type'].startswith('actual_'):
                # These need the stat feature which is already in base_features
                pass
        
        # Compute time-based features per entity
        if entity_to_windows:
            if verbose:
                print(f"  Computing time-based features for {len(entity_to_windows)} entities...")
            
            for entity, windows in tqdm(entity_to_windows.items(), desc="  Time features", disable=not verbose):
                if entity not in df_current.columns:
                    continue
                time_feats = compute_time_based_features_efficient(
                    df_historical, df_current, entity, sorted(windows)
                )
                all_features.update(time_feats)
        
        # Compute variability features per entity
        if entity_to_variability_windows:
            if verbose:
                print(f"  Computing variability features for {len(entity_to_variability_windows)} entities...")
            
            for entity, windows in tqdm(entity_to_variability_windows.items(), desc="  Variability", disable=not verbose):
                if entity not in df_current.columns:
                    continue
                var_feats = compute_variability_features_efficient(
                    df_historical, df_current, entity, sorted(windows)
                )
                all_features.update(var_feats)
        
        # Compute unique count features per entity pair
        if entity_pairs_to_windows:
            if verbose:
                print(f"  Computing unique count features for {len(entity_pairs_to_windows)} entity pairs...")
            
            for (entity_i, entity_j), windows in tqdm(entity_pairs_to_windows.items(), desc="  Unique counts", disable=not verbose):
                if entity_i not in df_current.columns or entity_j not in df_current.columns:
                    continue
                unique_feats = compute_unique_count_efficient(
                    df_historical, df_current, entity_i, entity_j, sorted(windows)
                )
                all_features.update(unique_feats)
        
        # Compute derived features from base features
        if verbose and requirements['derived_features']:
            print(f"  Computing {len(requirements['derived_features'])} derived features...")
        
        for feat_name, parsed in requirements['derived_features'].items():
            feat_type = parsed['type']
            
            # Velocity-to-recency ratio
            if feat_type == 'vdratio':
                count_vel_key = f"{parsed['entity']}_count_{parsed['short']}_by_{parsed['long']}"
                if count_vel_key not in all_features:
                    # Compute it
                    count_short = all_features.get(f"{parsed['entity']}_count_{parsed['short']}")
                    count_long = all_features.get(f"{parsed['entity']}_count_{parsed['long']}")
                    if count_short is not None and count_long is not None:
                        all_features[count_vel_key] = count_short / count_long / parsed['long']
                
                day_since = all_features.get(f"{parsed['entity']}_day_since")
                count_vel = all_features.get(count_vel_key)
                if day_since is not None and count_vel is not None:
                    all_features[feat_name] = count_vel / (day_since + 1)
            
            # Count velocity
            elif feat_type == 'count_velocity':
                count_short = all_features.get(f"{parsed['entity']}_count_{parsed['short']}")
                count_long = all_features.get(f"{parsed['entity']}_count_{parsed['long']}")
                if count_short is not None and count_long is not None:
                    all_features[feat_name] = count_short / count_long / parsed['long']
            
            # Count velocity squared
            elif feat_type == 'count_velocity_sq':
                count_short = all_features.get(f"{parsed['entity']}_count_{parsed['short']}")
                count_long = all_features.get(f"{parsed['entity']}_count_{parsed['long']}")
                if count_short is not None and count_long is not None:
                    all_features[feat_name] = count_short / count_long / (parsed['long'] ** 2)
            
            # Total amount velocity
            elif feat_type == 'total_velocity':
                total_short = all_features.get(f"{parsed['entity']}_total_{parsed['short']}")
                total_long = all_features.get(f"{parsed['entity']}_total_{parsed['long']}")
                if total_short is not None and total_long is not None:
                    all_features[feat_name] = total_short / total_long / parsed['long']
            
            # Actual/stat ratios
            elif feat_type.startswith('actual_'):
                stat_key = f"{parsed['entity']}_{parsed['stat']}_{parsed['window']}"
                stat_val = all_features.get(stat_key)
                if stat_val is not None:
                    all_features[feat_name] = df_current['Amount'] / stat_val
    
    # Step 3: Business logic features
    if requirements['business_features']:
        if verbose:
            print(f"  Computing {len(requirements['business_features'])} business logic features...")
        
        business_feats = compute_business_logic_features(
            df_historical, df_current, requirements['business_features'],
            zip_coords, us_zip_set
        )
        all_features.update(business_feats)
    
    # Convert to DataFrame with only selected features
    features_df = pd.DataFrame(index=df_current.index)
    
    for feat in selected_features:
        if feat in all_features:
            features_df[feat] = all_features[feat]
        else:
            if verbose:
                print(f"  Warning: Could not compute '{feat}'")
            features_df[feat] = 0
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    if verbose:
        print(f"Complete: {features_df.shape[1]} features constructed")
    
    return features_df


def load_zip_coordinates(zip_file_path):
    """
    Load ZIP code coordinates from CSV file.
    
    Args:
        zip_file_path: Path to zip_code_database.csv
    
    Returns:
        Dictionary mapping ZIP codes to {'latitude': lat, 'longitude': lon}
    """
    zip_df = pd.read_csv(zip_file_path)
    zip_df['zip'] = zip_df['zip'].astype(str)
    zip_to_coords = zip_df.set_index('zip')[['latitude', 'longitude']].to_dict('index')
    return zip_to_coords


def get_us_zip_set(zip_file_path):
    """
    Get set of valid US ZIP codes.
    
    Args:
        zip_file_path: Path to zip_code_database.csv
    
    Returns:
        Set of ZIP code strings
    """
    zip_df = pd.read_csv(zip_file_path)
    return set(zip_df['zip'].astype(str).values)
