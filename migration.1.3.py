import json
import os
import sys
import pyodbc
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from dataclasses import dataclass
import time

# Force UTF-8 encoding for console output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'migration_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MigrationStats:
    """Statistics tracking for migration process"""
    table_name: str
    total_records: int = 0
    migrated_records: int = 0
    failed_records: int = 0
    skipped_records: int = 0  # New field for duplicate records
    start_time: datetime = None
    end_time: datetime = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return (self.migrated_records / self.total_records) * 100
    
    @property
    def duration(self) -> str:
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return str(delta)
        return "N/A"

class DatabaseConfig:
    """Database connection configuration"""
    def __init__(self, server: str, database: str, username: str = None, 
                 password: str = None, trusted_connection: bool = True,
                 connection_timeout: int = 30):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.trusted_connection = trusted_connection
        self.connection_timeout = connection_timeout

class DataMigrator:
    """Main class for handling data migration between databases"""
    
    def __init__(self, source_config: DatabaseConfig, target_config: DatabaseConfig, 
                 handle_duplicates: str = "skip"):
        """
        Initialize DataMigrator
        
        Args:
            handle_duplicates: How to handle duplicate records
                - "skip": Skip duplicate records (default)
                - "update": Update existing records
                - "truncate": Clear target table before migration
        """
        self.source_config = source_config
        self.target_config = target_config
        self.source_connection = None
        self.target_connection = None
        self.migration_stats: List[MigrationStats] = []
        self.handle_duplicates = handle_duplicates
        
    def connect_databases(self) -> bool:
        """Establish connections to both source and target databases"""
        try:
            # Connect to source database
            logger.info(f"Connecting to source database: {self.source_config.database}")
            if self.source_config.trusted_connection:
                source_conn_str = (f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                                 f"SERVER={self.source_config.server};"
                                 f"DATABASE={self.source_config.database};"
                                 f"Trusted_Connection=yes;"
                                 f"Connection Timeout={self.source_config.connection_timeout};")
            else:
                source_conn_str = (f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                                 f"SERVER={self.source_config.server};"
                                 f"DATABASE={self.source_config.database};"
                                 f"UID={self.source_config.username};"
                                 f"PWD={self.source_config.password};"
                                 f"Connection Timeout={self.source_config.connection_timeout};")
            
            self.source_connection = pyodbc.connect(source_conn_str)
            logger.info("[SUCCESS] Source database connected successfully")
            
            # Connect to target database
            logger.info(f"Connecting to target database: {self.target_config.database}")
            if self.target_config.trusted_connection:
                target_conn_str = (f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                                 f"SERVER={self.target_config.server};"
                                 f"DATABASE={self.target_config.database};"
                                 f"Trusted_Connection=yes;"
                                 f"Connection Timeout={self.target_config.connection_timeout};")
            else:
                target_conn_str = (f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                                 f"SERVER={self.target_config.server};"
                                 f"DATABASE={self.target_config.database};"
                                 f"UID={self.target_config.username};"
                                 f"PWD={self.target_config.password};"
                                 f"Connection Timeout={self.target_config.connection_timeout};")
            
            self.target_connection = pyodbc.connect(target_conn_str)
            logger.info("[SUCCESS] Target database connected successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to databases: {str(e)}")
            return False
    
    def load_mapping_file(self, mapping_file_path: str) -> Optional[Dict]:
        """Load column mapping from JSON file"""
        try:
            with open(mapping_file_path, 'r') as f:
                mapping_data = json.load(f)
            
            logger.info(f"[SUCCESS] Loaded mapping file: {mapping_file_path}")
            logger.info(f"   Source table: {mapping_data.get('source_table', 'Unknown')}")
            logger.info(f"   Target table: {mapping_data.get('target_table', 'Unknown')}")
            logger.info(f"   Total mappings: {mapping_data.get('total_mappings', 0)}")
            
            return mapping_data
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to load mapping file {mapping_file_path}: {str(e)}")
            return None
    
    def build_column_mapping_dict(self, mappings: List[Dict]) -> Dict[str, str]:
        """Build a simple source->target column mapping dictionary"""
        column_map = {}
        
        for mapping in mappings:
            source_col = mapping.get('Source_Column')
            target_col = mapping.get('Target_Column')
            similarity = mapping.get('Similarity', 0)
            
            # Only include mappings with reasonable confidence
            if source_col and target_col and similarity >= 0.7:
                column_map[source_col] = target_col
                logger.debug(f"   Mapping: {source_col} -> {target_col} (confidence: {similarity})")
        
        return column_map
    
    def get_record_count(self, table_name: str, schema: str = "dbo", connection=None) -> int:
        """Get total record count for a table"""
        try:
            cursor = connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM [{schema}].[{table_name}]")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            logger.error(f"Failed to get record count for {table_name}: {str(e)}")
            return 0
    
    def get_primary_key_columns(self, table_name: str, schema: str = "dbo") -> List[str]:
        """Get primary key column names for a table"""
        try:
            cursor = self.target_connection.cursor()
            query = """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_SCHEMA + '.' + CONSTRAINT_NAME), 'IsPrimaryKey') = 1
            AND TABLE_NAME = ? AND TABLE_SCHEMA = ?
            ORDER BY ORDINAL_POSITION
            """
            cursor.execute(query, (table_name, schema))
            pk_columns = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return pk_columns
        except Exception as e:
            logger.error(f"Failed to get primary key columns for {table_name}: {str(e)}")
            return []
    
    def clear_target_table(self, table_name: str, schema: str = "dbo") -> bool:
        """Clear target table if requested"""
        try:
            cursor = self.target_connection.cursor()
            cursor.execute(f"TRUNCATE TABLE [{schema}].[{table_name}]")
            self.target_connection.commit()
            cursor.close()
            logger.info(f"[SUCCESS] Target table {table_name} cleared")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to clear target table {table_name}: {str(e)}")
            return False
    
    def record_exists(self, table_name: str, pk_columns: List[str], pk_values: List, 
                     schema: str = "dbo") -> bool:
        """Check if a record with given primary key values exists"""
        if not pk_columns or not pk_values:
            return False
        
        try:
            cursor = self.target_connection.cursor()
            where_clause = " AND ".join([f"[{col}] = ?" for col in pk_columns])
            query = f"SELECT COUNT(*) FROM [{schema}].[{table_name}] WHERE {where_clause}"
            cursor.execute(query, pk_values[:len(pk_columns)])
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except Exception as e:
            logger.debug(f"Error checking record existence: {str(e)}")
            return False
    
    def migrate_table_data(self, mapping_data: Dict, batch_size: int = 1000, 
                          validate_data: bool = True, schema: str = "dbo") -> MigrationStats:
        """Migrate data for a single table based on mapping configuration"""
        
        source_table = mapping_data['source_table']
        target_table = mapping_data['target_table']
        mappings = mapping_data['mappings']
        
        stats = MigrationStats(table_name=f"{source_table} -> {target_table}")
        stats.start_time = datetime.now()
        
        logger.info(f"[START] Starting migration: {source_table} -> {target_table}")
        
        try:
            # Handle table clearing if requested
            if self.handle_duplicates == "truncate":
                logger.info(f"[INFO] Clearing target table {target_table}")
                self.clear_target_table(target_table, schema)
            
            # Build column mapping
            column_map = self.build_column_mapping_dict(mappings)
            
            if not column_map:
                logger.error(f"[ERROR] No valid column mappings found for {source_table}")
                stats.errors.append("No valid column mappings found")
                return stats
            
            logger.info(f"[INFO] Column mappings ({len(column_map)} columns):")
            for src, tgt in column_map.items():
                logger.info(f"   {src} -> {tgt}")
            
            # Get primary key columns for duplicate handling
            pk_columns = self.get_primary_key_columns(target_table, schema)
            if pk_columns and self.handle_duplicates in ["skip", "update"]:
                logger.info(f"[INFO] Primary key columns for duplicate handling: {pk_columns}")
            
            # Get total record count
            stats.total_records = self.get_record_count(source_table, schema, self.source_connection)
            logger.info(f"[INFO] Total records to migrate: {stats.total_records:,}")
            
            if stats.total_records == 0:
                logger.warning(f"[WARNING] No records found in source table {source_table}")
                return stats
            
            # Build SQL queries
            source_columns = list(column_map.keys())
            target_columns = list(column_map.values())
            
            select_sql = f"SELECT {', '.join([f'[{col}]' for col in source_columns])} FROM [{schema}].[{source_table}]"
            insert_sql = f"INSERT INTO [{schema}].[{target_table}] ({', '.join([f'[{col}]' for col in target_columns])}) VALUES ({', '.join(['?' for _ in target_columns])})"
            
            logger.info(f"[INFO] Source query: {select_sql}")
            logger.info(f"[INFO] Target query: {insert_sql}")
            
            # Process data in batches
            source_cursor = self.source_connection.cursor()
            target_cursor = self.target_connection.cursor()
            
            # Enable fast execute many for better performance
            target_cursor.fast_executemany = True
            
            offset = 0
            batch_number = 1
            
            while offset < stats.total_records:
                logger.info(f"[BATCH] Processing batch {batch_number} (records {offset + 1} to {min(offset + batch_size, stats.total_records)})")
                
                # Fetch batch from source
                batch_sql = f"{select_sql} ORDER BY (SELECT NULL) OFFSET {offset} ROWS FETCH NEXT {batch_size} ROWS ONLY"
                source_cursor.execute(batch_sql)
                batch_data = source_cursor.fetchall()
                
                if not batch_data:
                    break
                
                # Prepare data for insertion
                insert_data = []
                batch_errors = 0
                batch_skipped = 0
                
                for row in batch_data:
                    try:
                        # Convert row to list and handle None values
                        row_data = []
                        for i, value in enumerate(row):
                            if value is None:
                                row_data.append(None)
                            else:
                                # Handle potential data type conversions
                                row_data.append(value)
                        
                        # Check for duplicates if primary key columns are available
                        if pk_columns and self.handle_duplicates == "skip":
                            # Get primary key values from the row
                            pk_values = []
                            for pk_col in pk_columns:
                                if pk_col in target_columns:
                                    pk_index = target_columns.index(pk_col)
                                    pk_values.append(row_data[pk_index])
                            
                            if pk_values and self.record_exists(target_table, pk_columns, pk_values, schema):
                                batch_skipped += 1
                                continue
                        
                        insert_data.append(tuple(row_data))
                        
                    except Exception as e:
                        batch_errors += 1
                        logger.warning(f"[WARNING] Error preparing row data: {str(e)}")
                        stats.errors.append(f"Row preparation error: {str(e)}")
                
                # Insert batch into target
                if insert_data:
                    try:
                        if self.handle_duplicates == "skip":
                            # Insert records one by one to handle individual duplicates
                            successful_inserts = 0
                            for record in insert_data:
                                try:
                                    target_cursor.execute(insert_sql, record)
                                    successful_inserts += 1
                                except pyodbc.IntegrityError as e:
                                    if "PRIMARY KEY constraint" in str(e) or "duplicate key" in str(e):
                                        batch_skipped += 1
                                    else:
                                        batch_errors += 1
                                        logger.warning(f"[WARNING] Integrity error: {str(e)}")
                                except Exception as e:
                                    batch_errors += 1
                                    logger.warning(f"[WARNING] Insert error: {str(e)}")
                            
                            self.target_connection.commit()
                            stats.migrated_records += successful_inserts
                        else:
                            # Use batch insert for better performance when not handling duplicates
                            target_cursor.executemany(insert_sql, insert_data)
                            self.target_connection.commit()
                            successful_inserts = len(insert_data)
                            stats.migrated_records += successful_inserts
                        
                        logger.info(f"[SUCCESS] Batch {batch_number} completed: {successful_inserts} records migrated, {batch_skipped} skipped")
                        
                    except Exception as e:
                        self.target_connection.rollback()
                        batch_errors += len(insert_data)
                        logger.error(f"[ERROR] Failed to insert batch {batch_number}: {str(e)}")
                        stats.errors.append(f"Batch {batch_number} insertion failed: {str(e)}")
                
                stats.failed_records += batch_errors
                stats.skipped_records += batch_skipped
                offset += batch_size
                batch_number += 1
                
                # Progress update
                processed = stats.migrated_records + stats.failed_records + stats.skipped_records
                progress = processed / stats.total_records * 100
                logger.info(f"[PROGRESS] {progress:.1f}% ({stats.migrated_records:,} migrated, {stats.skipped_records:,} skipped, {stats.failed_records:,} failed)")
            
            source_cursor.close()
            target_cursor.close()
            
            # Validation
            if validate_data:
                logger.info("[VALIDATION] Validating migration...")
                target_count = self.get_record_count(target_table, schema, self.target_connection)
                logger.info(f"[INFO] Target table now has {target_count:,} records")
                
                expected_count = stats.migrated_records
                if self.handle_duplicates == "truncate":
                    expected_count = stats.migrated_records
                
                if target_count < expected_count:
                    logger.warning(f"[WARNING] Record count lower than expected: Expected {expected_count}, Found {target_count}")
                    stats.errors.append(f"Record count mismatch: Expected {expected_count}, Found {target_count}")
            
        except Exception as e:
            logger.error(f"[ERROR] Migration failed for {source_table}: {str(e)}")
            stats.errors.append(f"Migration failed: {str(e)}")
        
        finally:
            stats.end_time = datetime.now()
            
        logger.info(f"[COMPLETE] Migration completed for {source_table} -> {target_table}")
        logger.info(f"   Duration: {stats.duration}")
        logger.info(f"   Success rate: {stats.success_rate:.1f}%")
        logger.info(f"   Total records: {stats.total_records:,}")
        logger.info(f"   Migrated: {stats.migrated_records:,}")
        logger.info(f"   Skipped: {stats.skipped_records:,}")
        logger.info(f"   Failed: {stats.failed_records:,}")
        
        return stats
    
    def migrate_all_mappings(self, mapping_directory: str = ".", batch_size: int = 1000, 
                           validate_data: bool = True, schema: str = "dbo") -> List[MigrationStats]:
        """Migrate all tables based on mapping files in the directory"""
        
        logger.info("[START] Starting batch migration process...")
        
        # Find all mapping files
        mapping_files = list(Path(mapping_directory).glob("mapping_*.json"))
        
        if not mapping_files:
            logger.error("[ERROR] No mapping files found in the directory")
            return []
        
        logger.info(f"[INFO] Found {len(mapping_files)} mapping files:")
        for file in mapping_files:
            logger.info(f"   - {file.name}")
        
        all_stats = []
        
        for mapping_file in mapping_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {mapping_file.name}")
            logger.info(f"{'='*60}")
            
            # Load mapping
            mapping_data = self.load_mapping_file(str(mapping_file))
            if not mapping_data:
                continue
            
            # Migrate table
            stats = self.migrate_table_data(mapping_data, batch_size, validate_data, schema)
            all_stats.append(stats)
            self.migration_stats.append(stats)
        
        return all_stats
    
    def generate_migration_report(self, output_file: str = None) -> str:
        """Generate a comprehensive migration report"""
        if not output_file:
            output_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        total_records = sum(stat.total_records for stat in self.migration_stats)
        total_migrated = sum(stat.migrated_records for stat in self.migration_stats)
        total_failed = sum(stat.failed_records for stat in self.migration_stats)
        total_skipped = sum(stat.skipped_records for stat in self.migration_stats)
        
        overall_success_rate = (total_migrated / total_records * 100) if total_records > 0 else 0
        
        report = []
        report.append("="*80)
        report.append("DATABASE MIGRATION REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Duplicate Handling Mode: {self.handle_duplicates}")
        report.append(f"Total Tables Processed: {len(self.migration_stats)}")
        report.append(f"Overall Success Rate: {overall_success_rate:.2f}%")
        report.append(f"Total Records: {total_records:,}")
        report.append(f"Successfully Migrated: {total_migrated:,}")
        report.append(f"Skipped (Duplicates): {total_skipped:,}")
        report.append(f"Failed: {total_failed:,}")
        report.append("")
        
        report.append("TABLE-BY-TABLE BREAKDOWN:")
        report.append("-" * 80)
        
        for i, stats in enumerate(self.migration_stats, 1):
            report.append(f"\n{i}. {stats.table_name}")
            report.append(f"   Duration: {stats.duration}")
            report.append(f"   Total Records: {stats.total_records:,}")
            report.append(f"   Migrated: {stats.migrated_records:,}")
            report.append(f"   Skipped: {stats.skipped_records:,}")
            report.append(f"   Failed: {stats.failed_records:,}")
            report.append(f"   Success Rate: {stats.success_rate:.2f}%")
            
            if stats.errors:
                report.append(f"   Errors ({len(stats.errors)}):")
                for error in stats.errors[:5]:  # Show first 5 errors
                    report.append(f"     - {error}")
                if len(stats.errors) > 5:
                    report.append(f"     ... and {len(stats.errors) - 5} more errors")
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"[SUCCESS] Migration report saved to: {output_file}")
        return report_text
    
    def disconnect(self):
        """Close database connections"""
        if self.source_connection:
            self.source_connection.close()
            logger.info("[SUCCESS] Source database connection closed")
        
        if self.target_connection:
            self.target_connection.close()
            logger.info("[SUCCESS] Target database connection closed")

def main():
    """Main function to run the migration process"""
    
    # Database configurations
    source_config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS",
        database="ax_db",
        trusted_connection=True
    )
    
    target_config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS", 
        database="sap_db",
        trusted_connection=True
    )
    
    # Initialize migrator with duplicate handling
    # Options: "skip", "update", "truncate"
    migrator = DataMigrator(source_config, target_config, handle_duplicates="skip")
    
    try:
        # Connect to databases
        if not migrator.connect_databases():
            logger.error("Failed to connect to databases. Exiting...")
            return
        
        # Run migration for all mapping files
        logger.info("[START] Starting automated migration process...")
        
        stats = migrator.migrate_all_mappings(
            mapping_directory=".",  # Current directory
            batch_size=1000,        # Process 1000 records at a time
            validate_data=True,     # Validate record counts after migration
            schema="dbo"           # Database schema
        )
        
        # Generate report
        if stats:
            report = migrator.generate_migration_report()
            logger.info("\n" + "="*60)
            logger.info("[COMPLETE] MIGRATION PROCESS COMPLETED!")
            logger.info("="*60)
            logger.info(f"Tables processed: {len(stats)}")
            
            successful_tables = [s for s in stats if s.success_rate > 90]
            logger.info(f"Successful migrations: {len(successful_tables)}")
            
            if len(successful_tables) < len(stats):
                logger.warning(f"[WARNING] Some migrations had issues. Check the detailed report.")
        else:
            logger.error("[ERROR] No migrations were completed successfully")
    
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error during migration: {str(e)}")
    
    finally:
        # Clean up connections
        migrator.disconnect()

def migrate_single_table(mapping_file_path: str, batch_size: int = 1000, handle_duplicates: str = "skip"):
    """Migrate a single table using a specific mapping file"""
    
    # Database configurations
    source_config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS",
        database="ax_db", 
        trusted_connection=True
    )
    
    target_config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS",
        database="sap_db",
        trusted_connection=True
    )
    
    migrator = DataMigrator(source_config, target_config, handle_duplicates=handle_duplicates)
    
    try:
        if not migrator.connect_databases():
            logger.error("Failed to connect to databases")
            return None
        
        # Load and migrate single table
        mapping_data = migrator.load_mapping_file(mapping_file_path)
        if mapping_data:
            stats = migrator.migrate_table_data(mapping_data, batch_size)
            migrator.migration_stats.append(stats)
            
            # Generate report for single table
            report = migrator.generate_migration_report()
            return stats
        
    except Exception as e:
        logger.error(f"Single table migration failed: {str(e)}")
        return None
    
    finally:
        migrator.disconnect()

if __name__ == "__main__":
    # Choose migration mode:
    
    # Option 1: Migrate all tables (recommended)
    main()
    
    # Option 2: Migrate single table with duplicate handling
    # migrate_single_table("mapping_Customers_AX_to_BusinessPartner_SAP.json", handle_duplicates="skip")
    
    # Option 3: Clear target table and migrate (use with caution!)
    #migrate_single_table("mapping_Customers_AX_to_BusinessPartner_SAP.json", handle_duplicates="truncate")
    
    logger.info("[COMPLETE] Migration script execution completed!")