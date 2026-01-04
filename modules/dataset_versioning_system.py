"""
Dataset Versioning and Lineage Tracking System
Part of SecureBank Phase 4: Enhanced Dataset Generation
"""

import pandas as pd
import numpy as np
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata"""
    dataset_id: str
    version: str
    name: str
    created_at: str
    created_by: str
    file_path: str
    file_size_bytes: int
    row_count: int
    column_count: int
    data_hash: str
    schema_hash: str
    quality_score: float
    drift_score: float
    parent_datasets: List[str] = field(default_factory=list)
    transformations_applied: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
@dataclass 
class DatasetLineage:
    """Dataset lineage tracking"""
    dataset_id: str
    lineage_type: str  # 'source', 'derived', 'merged', 'transformed'
    parent_datasets: List[str]
    transformation_pipeline: List[Dict[str, Any]]
    data_sources: List[str]
    processing_timestamp: str
    processing_duration_seconds: float
    quality_checks_passed: bool
    validation_results: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetVersion:
    """Version information for datasets"""
    version_id: str
    dataset_id: str
    version_number: str
    previous_version: Optional[str]
    changes_summary: str
    change_type: str  # 'major', 'minor', 'patch'
    compatibility_notes: str
    rollback_instructions: str

class DatasetVersioningSystem:
    """Advanced dataset versioning with full lineage tracking"""
    
    def __init__(self, storage_root: str = "securebank/storage"):
        self.storage_root = Path(storage_root)
        self.datasets_dir = self.storage_root / "datasets"
        self.metadata_dir = self.storage_root / "metadata"
        self.lineage_dir = self.storage_root / "lineage"
        self.versions_dir = self.storage_root / "versions"
        
        # Create directories
        for dir_path in [self.datasets_dir, self.metadata_dir, self.lineage_dir, self.versions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self.metadata_registry = self._load_metadata_registry()
        self.version_registry = self._load_version_registry()
        self.lineage_registry = self._load_lineage_registry()
        
    def create_dataset_version(self, df: pd.DataFrame, dataset_name: str,
                             transformation_info: Dict[str, Any] = None,
                             parent_datasets: List[str] = None,
                             tags: List[str] = None,
                             description: str = "",
                             version_type: str = "minor") -> DatasetMetadata:
        """
        Create a new versioned dataset with full tracking
        
        Args:
            df: Dataset dataframe
            dataset_name: Name of the dataset
            transformation_info: Information about transformations applied
            parent_datasets: List of parent dataset IDs
            tags: Dataset tags for categorization
            description: Dataset description
            version_type: Version increment type ('major', 'minor', 'patch')
        
        Returns:
            DatasetMetadata object
        """
        
        print(f"📦 Creating new dataset version: {dataset_name}")
        
        # Generate unique identifiers
        timestamp = datetime.now()
        dataset_id = self._generate_dataset_id(dataset_name, timestamp)
        
        # Determine version number
        version_number = self._get_next_version(dataset_name, version_type)
        
        # Calculate hashes
        data_hash = self._calculate_data_hash(df)
        schema_hash = self._calculate_schema_hash(df)
        
        # Save dataset file
        file_path = self.datasets_dir / f"{dataset_id}.csv"
        df.to_csv(file_path, index=False)
        file_size = file_path.stat().st_size
        
        # Create metadata
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            version=version_number,
            name=dataset_name,
            created_at=timestamp.isoformat(),
            created_by="securebank_system",
            file_path=str(file_path),
            file_size_bytes=file_size,
            row_count=len(df),
            column_count=len(df.columns),
            data_hash=data_hash,
            schema_hash=schema_hash,
            quality_score=0.0,  # To be updated by quality assessment
            drift_score=0.0,    # To be updated by drift detection
            parent_datasets=parent_datasets or [],
            transformations_applied=[transformation_info] if transformation_info else [],
            tags=tags or [],
            description=description
        )
        
        # Create lineage record
        lineage = DatasetLineage(
            dataset_id=dataset_id,
            lineage_type=self._determine_lineage_type(parent_datasets, transformation_info),
            parent_datasets=parent_datasets or [],
            transformation_pipeline=[transformation_info] if transformation_info else [],
            data_sources=self._extract_data_sources(parent_datasets),
            processing_timestamp=timestamp.isoformat(),
            processing_duration_seconds=0.0,  # To be updated
            quality_checks_passed=True,  # To be updated
            validation_results={}
        )
        
        # Create version record
        previous_version = self._get_latest_version(dataset_name)
        version_record = DatasetVersion(
            version_id=f"{dataset_id}_v{version_number}",
            dataset_id=dataset_id,
            version_number=version_number,
            previous_version=previous_version.dataset_id if previous_version else None,
            changes_summary=transformation_info.get('description', 'New dataset version') if transformation_info else 'Initial version',
            change_type=version_type,
            compatibility_notes="",
            rollback_instructions=""
        )
        
        # Save all records
        self._save_metadata(metadata)
        self._save_lineage(lineage)
        self._save_version(version_record)
        
        # Update registries
        self.metadata_registry[dataset_id] = metadata
        self.lineage_registry[dataset_id] = lineage
        self.version_registry[dataset_id] = version_record
        
        print(f"✅ Dataset version created: {dataset_id} (v{version_number})")
        return metadata
    
    def track_transformation(self, dataset_id: str, transformation: Dict[str, Any]) -> bool:
        """Track a transformation applied to a dataset"""
        
        if dataset_id not in self.metadata_registry:
            print(f"❌ Dataset {dataset_id} not found in registry")
            return False
        
        # Update metadata
        self.metadata_registry[dataset_id].transformations_applied.append(transformation)
        
        # Update lineage
        if dataset_id in self.lineage_registry:
            self.lineage_registry[dataset_id].transformation_pipeline.append(transformation)
        
        # Save updates
        self._save_metadata(self.metadata_registry[dataset_id])
        if dataset_id in self.lineage_registry:
            self._save_lineage(self.lineage_registry[dataset_id])
        
        print(f"✅ Transformation tracked for dataset {dataset_id}")
        return True
    
    def update_quality_metrics(self, dataset_id: str, quality_score: float, 
                             drift_score: float = None) -> bool:
        """Update quality and drift metrics for a dataset"""
        
        if dataset_id not in self.metadata_registry:
            return False
        
        metadata = self.metadata_registry[dataset_id]
        metadata.quality_score = quality_score
        
        if drift_score is not None:
            metadata.drift_score = drift_score
        
        self._save_metadata(metadata)
        print(f"✅ Quality metrics updated for dataset {dataset_id}")
        return True
    
    def get_dataset_lineage(self, dataset_id: str, depth: int = 10) -> Dict[str, Any]:
        """Get complete lineage for a dataset"""
        
        if dataset_id not in self.lineage_registry:
            return {"error": f"Dataset {dataset_id} not found"}
        
        lineage_tree = self._build_lineage_tree(dataset_id, depth, set())
        
        return {
            "dataset_id": dataset_id,
            "lineage_depth": depth,
            "lineage_tree": lineage_tree,
            "total_ancestors": self._count_ancestors(lineage_tree),
            "data_sources": self._extract_all_sources(lineage_tree)
        }
    
    def compare_dataset_versions(self, dataset_id_1: str, dataset_id_2: str) -> Dict[str, Any]:
        """Compare two dataset versions"""
        
        if dataset_id_1 not in self.metadata_registry or dataset_id_2 not in self.metadata_registry:
            return {"error": "One or both datasets not found"}
        
        meta1 = self.metadata_registry[dataset_id_1]
        meta2 = self.metadata_registry[dataset_id_2]
        
        # Schema comparison
        schema_changed = meta1.schema_hash != meta2.schema_hash
        data_changed = meta1.data_hash != meta2.data_hash
        
        # Size comparison
        size_change = meta2.file_size_bytes - meta1.file_size_bytes
        size_change_pct = (size_change / meta1.file_size_bytes) * 100 if meta1.file_size_bytes > 0 else 0
        
        row_change = meta2.row_count - meta1.row_count
        row_change_pct = (row_change / meta1.row_count) * 100 if meta1.row_count > 0 else 0
        
        column_change = meta2.column_count - meta1.column_count
        
        return {
            "dataset_1": {
                "id": dataset_id_1,
                "version": meta1.version,
                "created": meta1.created_at
            },
            "dataset_2": {
                "id": dataset_id_2, 
                "version": meta2.version,
                "created": meta2.created_at
            },
            "changes": {
                "schema_changed": schema_changed,
                "data_changed": data_changed,
                "size_change_bytes": size_change,
                "size_change_percentage": size_change_pct,
                "row_count_change": row_change,
                "row_count_change_percentage": row_change_pct,
                "column_count_change": column_change
            },
            "quality_comparison": {
                "quality_score_1": meta1.quality_score,
                "quality_score_2": meta2.quality_score,
                "quality_improvement": meta2.quality_score - meta1.quality_score,
                "drift_score_1": meta1.drift_score,
                "drift_score_2": meta2.drift_score
            }
        }
    
    def find_datasets_by_criteria(self, criteria: Dict[str, Any]) -> List[DatasetMetadata]:
        """Find datasets matching specific criteria"""
        
        matching_datasets = []
        
        for dataset_id, metadata in self.metadata_registry.items():
            match = True
            
            # Check each criterion
            for key, value in criteria.items():
                if key == 'name' and value.lower() not in metadata.name.lower():
                    match = False
                elif key == 'min_quality_score' and metadata.quality_score < value:
                    match = False
                elif key == 'max_drift_score' and metadata.drift_score > value:
                    match = False
                elif key == 'tags' and not any(tag in metadata.tags for tag in value):
                    match = False
                elif key == 'created_after':
                    created_date = datetime.fromisoformat(metadata.created_at)
                    if created_date < datetime.fromisoformat(value):
                        match = False
                elif key == 'min_rows' and metadata.row_count < value:
                    match = False
                elif key == 'max_rows' and metadata.row_count > value:
                    match = False
            
            if match:
                matching_datasets.append(metadata)
        
        # Sort by creation date (newest first)
        matching_datasets.sort(key=lambda x: x.created_at, reverse=True)
        
        return matching_datasets
    
    def cleanup_old_versions(self, dataset_name: str, keep_count: int = 5) -> int:
        """Clean up old dataset versions, keeping only the most recent"""
        
        # Find all versions of the dataset
        versions = []
        for dataset_id, metadata in self.metadata_registry.items():
            if metadata.name == dataset_name:
                versions.append((metadata.created_at, dataset_id))
        
        # Sort by creation date (newest first)
        versions.sort(reverse=True)
        
        # Keep only the specified number of versions
        if len(versions) <= keep_count:
            return 0
        
        versions_to_remove = versions[keep_count:]
        removed_count = 0
        
        for _, dataset_id in versions_to_remove:
            if self._remove_dataset_version(dataset_id):
                removed_count += 1
        
        print(f"🗑️ Cleaned up {removed_count} old versions of {dataset_name}")
        return removed_count
    
    def export_lineage_graph(self, dataset_id: str, format: str = "json") -> Union[str, Dict]:
        """Export lineage graph in various formats"""
        
        lineage_data = self.get_dataset_lineage(dataset_id)
        
        if format == "json":
            return json.dumps(lineage_data, indent=2, default=str)
        elif format == "mermaid":
            return self._generate_mermaid_graph(lineage_data)
        elif format == "dot":
            return self._generate_dot_graph(lineage_data)
        else:
            return lineage_data
    
    def generate_dataset_catalog(self, save_path: str = None) -> str:
        """Generate comprehensive dataset catalog"""
        
        catalog = f"""
# 📊 SECUREBANK DATASET CATALOG
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 CATALOG OVERVIEW
- **Total Datasets**: {len(self.metadata_registry)}
- **Total Versions**: {len(self.version_registry)}
- **Storage Used**: {self._calculate_total_storage():.2f} MB
- **Average Quality Score**: {self._calculate_average_quality():.2f}

## 📋 DATASET INVENTORY
"""
        
        # Group datasets by name
        dataset_groups = {}
        for metadata in self.metadata_registry.values():
            if metadata.name not in dataset_groups:
                dataset_groups[metadata.name] = []
            dataset_groups[metadata.name].append(metadata)
        
        for dataset_name, versions in dataset_groups.items():
            # Sort versions by creation date
            versions.sort(key=lambda x: x.created_at, reverse=True)
            latest = versions[0]
            
            catalog += f"""
### 📁 {dataset_name}
- **Latest Version**: {latest.version} ({latest.dataset_id})
- **Created**: {latest.created_at}
- **Size**: {latest.row_count:,} rows × {latest.column_count} columns ({latest.file_size_bytes / 1024 / 1024:.2f} MB)
- **Quality Score**: {latest.quality_score:.2f}
- **Drift Score**: {latest.drift_score:.2f}
- **Total Versions**: {len(versions)}
- **Tags**: {', '.join(latest.tags) if latest.tags else 'None'}
- **Description**: {latest.description or 'No description'}
"""
            
            if len(versions) > 1:
                catalog += f"- **Previous Versions**: "
                for version in versions[1:6]:  # Show up to 5 previous versions
                    catalog += f"{version.version} "
                if len(versions) > 6:
                    catalog += f"... ({len(versions) - 6} more)"
                catalog += "\n"
        
        catalog += f"""
## 🔗 LINEAGE SUMMARY
"""
        
        # Analyze lineage patterns
        source_datasets = 0
        derived_datasets = 0
        merged_datasets = 0
        
        for lineage in self.lineage_registry.values():
            if lineage.lineage_type == 'source':
                source_datasets += 1
            elif lineage.lineage_type == 'derived':
                derived_datasets += 1
            elif lineage.lineage_type == 'merged':
                merged_datasets += 1
        
        catalog += f"""
- **Source Datasets**: {source_datasets}
- **Derived Datasets**: {derived_datasets}
- **Merged Datasets**: {merged_datasets}
- **Complex Lineages**: {len([l for l in self.lineage_registry.values() if len(l.parent_datasets) > 1])}

## 🏷️ DATASET TAGS
"""
        
        # Aggregate all tags
        all_tags = {}
        for metadata in self.metadata_registry.values():
            for tag in metadata.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1
        
        sorted_tags = sorted(all_tags.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags:
            catalog += f"- **{tag}**: {count} datasets\n"
        
        catalog += f"""
## 📊 QUALITY METRICS
- **High Quality (>0.8)**: {len([m for m in self.metadata_registry.values() if m.quality_score > 0.8])} datasets
- **Medium Quality (0.6-0.8)**: {len([m for m in self.metadata_registry.values() if 0.6 <= m.quality_score <= 0.8])} datasets
- **Low Quality (<0.6)**: {len([m for m in self.metadata_registry.values() if m.quality_score < 0.6])} datasets

## ⚠️ DRIFT ALERTS
- **No Drift (<0.1)**: {len([m for m in self.metadata_registry.values() if m.drift_score < 0.1])} datasets
- **Low Drift (0.1-0.3)**: {len([m for m in self.metadata_registry.values() if 0.1 <= m.drift_score < 0.3])} datasets
- **High Drift (≥0.3)**: {len([m for m in self.metadata_registry.values() if m.drift_score >= 0.3])} datasets

## 🔧 MAINTENANCE RECOMMENDATIONS
"""
        
        # Generate maintenance recommendations
        high_drift_datasets = [m for m in self.metadata_registry.values() if m.drift_score >= 0.3]
        if high_drift_datasets:
            catalog += f"1. **High Priority**: {len(high_drift_datasets)} datasets with significant drift require attention\n"
        
        low_quality_datasets = [m for m in self.metadata_registry.values() if m.quality_score < 0.6]
        if low_quality_datasets:
            catalog += f"2. **Quality Issues**: {len(low_quality_datasets)} datasets need quality improvement\n"
        
        old_datasets = [m for m in self.metadata_registry.values() 
                       if (datetime.now() - datetime.fromisoformat(m.created_at)).days > 30]
        if old_datasets:
            catalog += f"3. **Aging Data**: {len(old_datasets)} datasets are over 30 days old\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(catalog)
            print(f"📄 Dataset catalog saved to: {save_path}")
        
        return catalog
    
    # Private helper methods
    def _generate_dataset_id(self, dataset_name: str, timestamp: datetime) -> str:
        """Generate unique dataset ID"""
        name_clean = "".join(c for c in dataset_name if c.isalnum() or c in "_-")
        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
        return f"{name_clean}_{timestamp_str}"
    
    def _get_next_version(self, dataset_name: str, version_type: str) -> str:
        """Calculate next version number"""
        existing_versions = []
        for metadata in self.metadata_registry.values():
            if metadata.name == dataset_name:
                existing_versions.append(metadata.version)
        
        if not existing_versions:
            return "1.0.0"
        
        # Parse latest version
        latest_version = max(existing_versions, key=lambda v: [int(x) for x in v.split('.')])
        major, minor, patch = map(int, latest_version.split('.'))
        
        if version_type == "major":
            return f"{major + 1}.0.0"
        elif version_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:  # patch
            return f"{major}.{minor}.{patch + 1}"
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataset content"""
        df_string = df.to_csv(index=False)
        return hashlib.md5(df_string.encode()).hexdigest()
    
    def _calculate_schema_hash(self, df: pd.DataFrame) -> str:
        """Calculate hash of dataset schema"""
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        schema_string = json.dumps(schema_info, sort_keys=True)
        return hashlib.md5(schema_string.encode()).hexdigest()
    
    def _determine_lineage_type(self, parent_datasets: List[str], transformation_info: Dict) -> str:
        """Determine the type of lineage"""
        if not parent_datasets:
            return 'source'
        elif len(parent_datasets) == 1:
            return 'derived'
        else:
            return 'merged'
    
    def _extract_data_sources(self, parent_datasets: List[str]) -> List[str]:
        """Extract original data sources from parent datasets"""
        sources = []
        for parent_id in parent_datasets or []:
            if parent_id in self.lineage_registry:
                lineage = self.lineage_registry[parent_id]
                if lineage.lineage_type == 'source':
                    sources.append(parent_id)
                else:
                    sources.extend(lineage.data_sources)
        return list(set(sources))
    
    def _get_latest_version(self, dataset_name: str) -> Optional[DatasetMetadata]:
        """Get latest version of a dataset"""
        versions = []
        for metadata in self.metadata_registry.values():
            if metadata.name == dataset_name:
                versions.append(metadata)
        
        if not versions:
            return None
        
        return max(versions, key=lambda x: x.created_at)
    
    def _build_lineage_tree(self, dataset_id: str, depth: int, visited: set) -> Dict[str, Any]:
        """Build lineage tree recursively"""
        if depth <= 0 or dataset_id in visited:
            return {"dataset_id": dataset_id, "truncated": True}
        
        visited.add(dataset_id)
        
        tree = {
            "dataset_id": dataset_id,
            "metadata": asdict(self.metadata_registry.get(dataset_id, {})),
            "lineage": asdict(self.lineage_registry.get(dataset_id, {})),
            "children": []
        }
        
        # Find parent datasets
        if dataset_id in self.lineage_registry:
            lineage = self.lineage_registry[dataset_id]
            for parent_id in lineage.parent_datasets:
                child_tree = self._build_lineage_tree(parent_id, depth - 1, visited.copy())
                tree["children"].append(child_tree)
        
        return tree
    
    def _count_ancestors(self, lineage_tree: Dict[str, Any]) -> int:
        """Count total ancestors in lineage tree"""
        count = 0
        for child in lineage_tree.get("children", []):
            count += 1 + self._count_ancestors(child)
        return count
    
    def _extract_all_sources(self, lineage_tree: Dict[str, Any]) -> List[str]:
        """Extract all data sources from lineage tree"""
        sources = []
        
        lineage_info = lineage_tree.get("lineage", {})
        if lineage_info.get("lineage_type") == "source":
            sources.append(lineage_tree["dataset_id"])
        
        for child in lineage_tree.get("children", []):
            sources.extend(self._extract_all_sources(child))
        
        return list(set(sources))
    
    def _remove_dataset_version(self, dataset_id: str) -> bool:
        """Remove a dataset version and all associated files"""
        try:
            # Remove metadata
            if dataset_id in self.metadata_registry:
                metadata = self.metadata_registry[dataset_id]
                
                # Remove dataset file
                if os.path.exists(metadata.file_path):
                    os.remove(metadata.file_path)
                
                # Remove metadata file
                metadata_file = self.metadata_dir / f"{dataset_id}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                # Remove from registry
                del self.metadata_registry[dataset_id]
            
            # Remove lineage
            if dataset_id in self.lineage_registry:
                lineage_file = self.lineage_dir / f"{dataset_id}.json"
                if lineage_file.exists():
                    lineage_file.unlink()
                del self.lineage_registry[dataset_id]
            
            # Remove version
            if dataset_id in self.version_registry:
                version_file = self.versions_dir / f"{dataset_id}.json"
                if version_file.exists():
                    version_file.unlink()
                del self.version_registry[dataset_id]
            
            return True
        except Exception as e:
            print(f"❌ Error removing dataset {dataset_id}: {str(e)}")
            return False
    
    def _generate_mermaid_graph(self, lineage_data: Dict[str, Any]) -> str:
        """Generate Mermaid graph representation of lineage"""
        def traverse_tree(node, graph_lines, node_id=0):
            current_id = f"node{node_id}"
            dataset_id = node["dataset_id"]
            
            graph_lines.append(f"    {current_id}[\"{dataset_id}\"]")
            
            child_id = node_id + 1
            for child in node.get("children", []):
                child_current_id = f"node{child_id}"
                graph_lines.append(f"    {child_current_id} --> {current_id}")
                child_id = traverse_tree(child, graph_lines, child_id)
            
            return child_id
        
        graph_lines = ["graph TD"]
        traverse_tree(lineage_data["lineage_tree"], graph_lines)
        
        return "\n".join(graph_lines)
    
    def _generate_dot_graph(self, lineage_data: Dict[str, Any]) -> str:
        """Generate Graphviz DOT format representation"""
        lines = ["digraph DatasetLineage {"]
        lines.append("    rankdir=TB;")
        lines.append("    node [shape=box];")
        
        def traverse_tree(node):
            dataset_id = node["dataset_id"]
            lines.append(f"    \"{dataset_id}\";")
            
            for child in node.get("children", []):
                child_id = child["dataset_id"]
                lines.append(f"    \"{child_id}\" -> \"{dataset_id}\";")
                traverse_tree(child)
        
        traverse_tree(lineage_data["lineage_tree"])
        lines.append("}")
        
        return "\n".join(lines)
    
    def _calculate_total_storage(self) -> float:
        """Calculate total storage used in MB"""
        total_bytes = sum(metadata.file_size_bytes for metadata in self.metadata_registry.values())
        return total_bytes / (1024 * 1024)
    
    def _calculate_average_quality(self) -> float:
        """Calculate average quality score across all datasets"""
        if not self.metadata_registry:
            return 0.0
        
        total_score = sum(metadata.quality_score for metadata in self.metadata_registry.values())
        return total_score / len(self.metadata_registry)
    
    def _save_metadata(self, metadata: DatasetMetadata) -> None:
        """Save metadata to file"""
        metadata_file = self.metadata_dir / f"{metadata.dataset_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def _save_lineage(self, lineage: DatasetLineage) -> None:
        """Save lineage to file"""
        lineage_file = self.lineage_dir / f"{lineage.dataset_id}.json"
        with open(lineage_file, 'w') as f:
            json.dump(asdict(lineage), f, indent=2, default=str)
    
    def _save_version(self, version: DatasetVersion) -> None:
        """Save version to file"""
        version_file = self.versions_dir / f"{version.dataset_id}.json"
        with open(version_file, 'w') as f:
            json.dump(asdict(version), f, indent=2, default=str)
    
    def _load_metadata_registry(self) -> Dict[str, DatasetMetadata]:
        """Load existing metadata registry"""
        registry = {}
        if self.metadata_dir.exists():
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    metadata = DatasetMetadata(**data)
                    registry[metadata.dataset_id] = metadata
                except Exception as e:
                    print(f"Warning: Could not load metadata from {metadata_file}: {e}")
        return registry
    
    def _load_lineage_registry(self) -> Dict[str, DatasetLineage]:
        """Load existing lineage registry"""
        registry = {}
        if self.lineage_dir.exists():
            for lineage_file in self.lineage_dir.glob("*.json"):
                try:
                    with open(lineage_file, 'r') as f:
                        data = json.load(f)
                    lineage = DatasetLineage(**data)
                    registry[lineage.dataset_id] = lineage
                except Exception as e:
                    print(f"Warning: Could not load lineage from {lineage_file}: {e}")
        return registry
    
    def _load_version_registry(self) -> Dict[str, DatasetVersion]:
        """Load existing version registry"""
        registry = {}
        if self.versions_dir.exists():
            for version_file in self.versions_dir.glob("*.json"):
                try:
                    with open(version_file, 'r') as f:
                        data = json.load(f)
                    version = DatasetVersion(**data)
                    registry[version.dataset_id] = version
                except Exception as e:
                    print(f"Warning: Could not load version from {version_file}: {e}")
        return registry

# Integration class for SecureBank system
class SecureBankDatasetManager:
    """Integration class for SecureBank dataset management"""
    
    def __init__(self, storage_root: str = "securebank/storage"):
        self.versioning_system = DatasetVersioningSystem(storage_root)
        
    def create_versioned_dataset(self, df: pd.DataFrame, dataset_name: str,
                                transformation_details: Dict[str, Any] = None,
                                quality_score: float = None,
                                drift_score: float = None) -> str:
        """Create a new versioned dataset with quality metrics"""
        
        # Create dataset version
        metadata = self.versioning_system.create_dataset_version(
            df=df,
            dataset_name=dataset_name,
            transformation_info=transformation_details,
            tags=["fraud_detection", "banking", "ml_training"]
        )
        
        # Update quality metrics if provided
        if quality_score is not None or drift_score is not None:
            self.versioning_system.update_quality_metrics(
                metadata.dataset_id, 
                quality_score or 0.0, 
                drift_score
            )
        
        return metadata.dataset_id
    
    def get_best_dataset_version(self, dataset_name: str, 
                                criteria: Dict[str, Any] = None) -> Optional[str]:
        """Get the best dataset version based on quality criteria"""
        
        default_criteria = {
            'name': dataset_name,
            'min_quality_score': 0.7,
            'max_drift_score': 0.3
        }
        
        search_criteria = {**default_criteria, **(criteria or {})}
        matching_datasets = self.versioning_system.find_datasets_by_criteria(search_criteria)
        
        if not matching_datasets:
            return None
        
        # Return the dataset with highest quality score
        best_dataset = max(matching_datasets, key=lambda x: x.quality_score)
        return best_dataset.dataset_id

# Example usage and testing
if __name__ == "__main__":
    # Initialize versioning system
    manager = SecureBankDatasetManager("test_storage")
    
    # Create sample datasets
    np.random.seed(42)
    
    # Original dataset
    original_data = {
        'trans_date_trans_time': pd.date_range('2023-01-01', periods=1000, freq='1H'),
        'cc_num': np.random.choice(range(1000, 2000), 1000),
        'merchant': np.random.choice(['Store_A', 'Store_B', 'Store_C'], 1000),
        'category': np.random.choice(['grocery_pos', 'gas_transport'], 1000),
        'amt': np.random.normal(50, 20, 1000),
        'is_fraud': np.random.choice([0, 1], 1000, p=[0.95, 0.05])
    }
    original_df = pd.DataFrame(original_data)
    
    # Create first version
    dataset_id_1 = manager.create_versioned_dataset(
        df=original_df,
        dataset_name="fraud_detection_dataset",
        transformation_details={
            "operation": "initial_creation",
            "description": "Initial fraud detection dataset",
            "parameters": {"source": "raw_banking_data"}
        },
        quality_score=0.85,
        drift_score=0.0
    )
    
    print(f"Created dataset version 1: {dataset_id_1}")
    
    # Create enhanced version with feature engineering
    enhanced_df = original_df.copy()
    enhanced_df['hour'] = enhanced_df['trans_date_trans_time'].dt.hour
    enhanced_df['is_weekend'] = enhanced_df['trans_date_trans_time'].dt.dayofweek >= 5
    enhanced_df['amount_log'] = np.log1p(enhanced_df['amt'])
    
    dataset_id_2 = manager.create_versioned_dataset(
        df=enhanced_df,
        dataset_name="fraud_detection_dataset",
        transformation_details={
            "operation": "feature_engineering",
            "description": "Added temporal and amount features",
            "parameters": {"features_added": ["hour", "is_weekend", "amount_log"]},
            "parent_dataset": dataset_id_1
        },
        quality_score=0.92,
        drift_score=0.15
    )
    
    print(f"Created dataset version 2: {dataset_id_2}")
    
    # Generate catalog
    catalog = manager.versioning_system.generate_dataset_catalog()
    print("\n" + "="*60)
    print(catalog[:2000] + "..." if len(catalog) > 2000 else catalog)