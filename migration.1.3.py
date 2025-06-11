import json
import pyodbc
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from tqdm import tqdm
import time
import sys
import io

# Fix Unicode encoding for Windows console
def setup_console_encoding():
    """Setup proper console encoding for Unicode characters"""
    if sys.platform.startswith('win'):
        # For Windows, set console to UTF-8
        try:
            # Try to set console code page to UTF-8
            import subprocess
            subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
        except:
            pass

# Configure logging with proper encoding
def setup_logging():
    """Setup logging with UTF-8 encoding to handle Unicode characters"""
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler('migration.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize console encoding and logging
setup_console_encoding()
logger = setup_logging()

@dataclass
class MigrationResult:
    """Data class to store migration results"""
    total_records: int = 0
    successful_records: int = 0
    failed_records: int = 0
    errors: List[str] = None
    start_time: datetime = None
    end_time: datetime = None
    duration_seconds: float = 0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class DatabaseConfig:
    """Database connection configuration"""
    def __init__(self, server: str, database: str, username: str = None, password: str = None, trusted_connection: bool = True):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.trusted_connection = trusted_connection

class DataMigrationEngine:
    """Complete data migration engine with validation and error handling"""
    
    def __init__(self, source_config: DatabaseConfig, target_config: DatabaseConfig):
        self.source_config = source_config
        self.target_config = target_config
        self.source_connection = None
        self.target_connection = None
        self.mappings = None
        
    def connect_databases(self) -> bool:
        """Connect to both source and target databases"""
        try:
            # Connect to source database
            if self.source_config.trusted_connection:
                source_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.source_config.server};DATABASE={self.source_config.database};Trusted_Connection=yes;"
            else:
                source_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.source_config.server};DATABASE={self.source_config.database};UID={self.source_config.username};PWD={self.source_config.password};"
            
            self.source_connection = pyodbc.connect(source_conn_str)
            logger.info(f"[OK] Connected to source database: {self.source_config.server}/{self.source_config.database}")
            
            # Connect to target database
            if self.target_config.trusted_connection:
                target_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.target_config.server};DATABASE={self.target_config.database};Trusted_Connection=yes;"
            else:
                target_conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.target_config.server};DATABASE={self.target_config.database};UID={self.target_config.username};PWD={self.target_config.password};"
            
            self.target_connection = pyodbc.connect(target_conn_str)
            logger.info(f"[OK] Connected to target database: {self.target_config.server}/{self.target_config.database}")
            
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to databases: {str(e)}")
            return False
    
    def load_mappings(self, mapping_file: str = "demo_column_mapping.json") -> bool:
        """Load column mappings from JSON file"""
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.mappings = json.load(f)
            logger.info(f"[OK] Loaded {len(self.mappings)} column mappings from {mapping_file}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to load mappings: {str(e)}")
            return False
    
    def validate_mappings(self, source_table: str, target_table: str, source_schema: str = "dbo", target_schema: str = "dbo") -> bool:
        """Validate that mapped columns exist in both tables"""
        try:
            source_cursor = self.source_connection.cursor()
            target_cursor = self.target_connection.cursor()
            
            # Get source table columns
            source_cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """, (source_schema, source_table))
            source_columns = {row[0]: row[1] for row in source_cursor.fetchall()}
            
            # Get target table columns
            target_cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """, (target_schema, target_table))
            target_columns = {row[0]: row[1] for row in target_cursor.fetchall()}
            
            # Validate mappings
            valid_mappings = []
            invalid_mappings = []
            
            for mapping in self.mappings:
                source_col = mapping['Source_Column']
                target_col = mapping['Target_Column']
                
                if source_col not in source_columns:
                    invalid_mappings.append(f"Source column '{source_col}' not found in {source_table}")
                elif target_col not in target_columns:
                    invalid_mappings.append(f"Target column '{target_col}' not found in {target_table}")
                else:
                    valid_mappings.append(mapping)
            
            if invalid_mappings:
                logger.warning(f"[WARNING] Found {len(invalid_mappings)} invalid mappings:")
                for invalid in invalid_mappings:
                    logger.warning(f"   - {invalid}")
            
            self.mappings = valid_mappings
            logger.info(f"[OK] Validated {len(valid_mappings)} column mappings")
            
            source_cursor.close()
            target_cursor.close()
            
            return len(valid_mappings) > 0
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to validate mappings: {str(e)}")
            return False
    
    def get_source_data_count(self, source_table: str, source_schema: str = "dbo", where_clause: str = None) -> int:
        """Get total count of records to migrate"""
        try:
            cursor = self.source_connection.cursor()
            
            sql = f"SELECT COUNT(*) FROM {source_schema}.{source_table}"
            if where_clause:
                sql += f" WHERE {where_clause}"
            
            cursor.execute(sql)
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to get source data count: {str(e)}")
            return 0
    
    def migrate_data_batch(self, source_table: str, target_table: str, 
                          source_schema: str = "dbo", target_schema: str = "dbo",
                          batch_size: int = 1000, where_clause: str = None,
                          truncate_target: bool = False) -> MigrationResult:
        """Migrate data in batches with progress tracking"""
        
        result = MigrationResult()
        result.start_time = datetime.now()
        
        try:
            # Truncate target table if requested
            if truncate_target:
                logger.info(f"[DELETE] Truncating target table {target_schema}.{target_table}")
                target_cursor = self.target_connection.cursor()
                target_cursor.execute(f"TRUNCATE TABLE {target_schema}.{target_table}")
                self.target_connection.commit()
                target_cursor.close()
            
            # Get total record count
            total_records = self.get_source_data_count(source_table, source_schema, where_clause)
            result.total_records = total_records
            
            if total_records == 0:
                logger.warning("[WARNING] No records found to migrate")
                return result
            
            logger.info(f"[DATA] Starting migration of {total_records:,} records in batches of {batch_size:,}")
            
            # Build column lists for SQL
            source_columns = [mapping['Source_Column'] for mapping in self.mappings]
            target_columns = [mapping['Target_Column'] for mapping in self.mappings]
            
            # Build SELECT query
            select_sql = f"""
            SELECT {', '.join(source_columns)}
            FROM {source_schema}.{source_table}
            """
            if where_clause:
                select_sql += f" WHERE {where_clause}"
            select_sql += f" ORDER BY (SELECT NULL) OFFSET ? ROWS FETCH NEXT {batch_size} ROWS ONLY"
            
            # Build INSERT query
            insert_sql = f"""
            INSERT INTO {target_schema}.{target_table} 
            ({', '.join(target_columns)})
            VALUES ({', '.join(['?' for _ in target_columns])})
            """
            
            # Process data in batches
            offset = 0
            progress_bar = tqdm(total=total_records, desc="Migrating records", unit="records")
            
            while offset < total_records:
                try:
                    # Fetch batch from source
                    source_cursor = self.source_connection.cursor()
                    source_cursor.execute(select_sql, (offset,))
                    batch_data = source_cursor.fetchall()
                    source_cursor.close()
                    
                    if not batch_data:
                        break
                    
                    # Insert batch to target
                    target_cursor = self.target_connection.cursor()
                    
                    for row in batch_data:
                        try:
                            # Convert row to list and handle None values
                            row_data = [None if val is None else val for val in row]
                            target_cursor.execute(insert_sql, row_data)
                            result.successful_records += 1
                            
                        except Exception as row_error:
                            result.failed_records += 1
                            error_msg = f"Row error at offset {offset}: {str(row_error)}"
                            result.errors.append(error_msg)
                            logger.warning(f"[WARNING] {error_msg}")
                    
                    # Commit batch
                    self.target_connection.commit()
                    target_cursor.close()
                    
                    # Update progress
                    batch_processed = len(batch_data)
                    progress_bar.update(batch_processed)
                    offset += batch_size
                    
                    # Log progress every 10 batches
                    if (offset // batch_size) % 10 == 0:
                        logger.info(f"[PROGRESS] Processed {result.successful_records:,} records ({(result.successful_records/total_records)*100:.1f}%)")
                
                except Exception as batch_error:
                    error_msg = f"Batch error at offset {offset}: {str(batch_error)}"
                    result.errors.append(error_msg)
                    logger.error(f"[ERROR] {error_msg}")
                    break
            
            progress_bar.close()
            
        except Exception as e:
            error_msg = f"Migration failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(f"[ERROR] {error_msg}")
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def migrate_data_pandas(self, source_table: str, target_table: str,
                           source_schema: str = "dbo", target_schema: str = "dbo",
                           chunk_size: int = 10000, where_clause: str = None) -> MigrationResult:
        """Alternative migration using pandas for better performance with large datasets"""
        
        result = MigrationResult()
        result.start_time = datetime.now()
        
        try:
            # Build column mapping dictionary
            column_mapping = {mapping['Source_Column']: mapping['Target_Column'] for mapping in self.mappings}
            source_columns = list(column_mapping.keys())
            
            # Build query
            query = f"SELECT {', '.join(source_columns)} FROM {source_schema}.{source_table}"
            if where_clause:
                query += f" WHERE {where_clause}"
            
            logger.info(f"[DATA] Starting pandas-based migration with chunk size {chunk_size:,}")
            
            # Read and migrate in chunks
            chunk_count = 0
            for chunk_df in pd.read_sql(query, self.source_connection, chunksize=chunk_size):
                try:
                    # Rename columns according to mapping
                    chunk_df = chunk_df.rename(columns=column_mapping)
                    
                    # Write to target database
                    chunk_df.to_sql(
                        name=target_table,
                        con=self.target_connection,
                        schema=target_schema,
                        if_exists='append',
                        index=False,
                        method='multi'
                    )
                    
                    chunk_count += 1
                    records_in_chunk = len(chunk_df)
                    result.successful_records += records_in_chunk
                    result.total_records += records_in_chunk
                    
                    logger.info(f"[PROGRESS] Processed chunk {chunk_count}: {records_in_chunk:,} records")
                    
                except Exception as chunk_error:
                    error_msg = f"Chunk {chunk_count} error: {str(chunk_error)}"
                    result.errors.append(error_msg)
                    logger.error(f"[ERROR] {error_msg}")
                    result.failed_records += len(chunk_df) if 'chunk_df' in locals() else chunk_size
            
        except Exception as e:
            error_msg = f"Pandas migration failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(f"[ERROR] {error_msg}")
        
        finally:
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
        
        return result
    
    def validate_migration(self, source_table: str, target_table: str,
                          source_schema: str = "dbo", target_schema: str = "dbo") -> Dict:
        """Validate migration by comparing record counts and sample data"""
        
        validation_result = {
            'source_count': 0,
            'target_count': 0,
            'count_match': False,
            'sample_validation': [],
            'errors': []
        }
        
        try:
            # Compare record counts
            source_cursor = self.source_connection.cursor()
            source_cursor.execute(f"SELECT COUNT(*) FROM {source_schema}.{source_table}")
            validation_result['source_count'] = source_cursor.fetchone()[0]
            source_cursor.close()
            
            target_cursor = self.target_connection.cursor()
            target_cursor.execute(f"SELECT COUNT(*) FROM {target_schema}.{target_table}")
            validation_result['target_count'] = target_cursor.fetchone()[0]
            target_cursor.close()
            
            validation_result['count_match'] = validation_result['source_count'] == validation_result['target_count']
            
            # Sample validation (compare first 5 records)
            source_columns = [mapping['Source_Column'] for mapping in self.mappings[:3]]  # First 3 columns
            target_columns = [mapping['Target_Column'] for mapping in self.mappings[:3]]
            
            source_cursor = self.source_connection.cursor()
            source_cursor.execute(f"SELECT TOP 5 {', '.join(source_columns)} FROM {source_schema}.{source_table}")
            source_sample = source_cursor.fetchall()
            source_cursor.close()
            
            target_cursor = self.target_connection.cursor()
            target_cursor.execute(f"SELECT TOP 5 {', '.join(target_columns)} FROM {target_schema}.{target_table}")
            target_sample = target_cursor.fetchall()
            target_cursor.close()
            
            # Compare samples
            for i, (source_row, target_row) in enumerate(zip(source_sample, target_sample)):
                sample_match = list(source_row) == list(target_row)
                validation_result['sample_validation'].append({
                    'row': i + 1,
                    'match': sample_match,
                    'source': list(source_row),
                    'target': list(target_row)
                })
            
        except Exception as e:
            validation_result['errors'].append(str(e))
            logger.error(f"[ERROR] Validation error: {str(e)}")
        
        return validation_result
    
    def generate_migration_report(self, result: MigrationResult, 
                                 validation_result: Dict = None) -> str:
        """Generate comprehensive migration report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("[START] DATA MIGRATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Migration Date: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Duration: {result.duration_seconds:.2f} seconds")
        report_lines.append("")
        
        # Migration Statistics
        report_lines.append("[DATA] MIGRATION STATISTICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Records: {result.total_records:,}")
        report_lines.append(f"Successfully Migrated: {result.successful_records:,}")
        report_lines.append(f"Failed Records: {result.failed_records:,}")
        
        if result.total_records > 0:
            success_rate = (result.successful_records / result.total_records) * 100
            report_lines.append(f"Success Rate: {success_rate:.2f}%")
            
            if result.duration_seconds > 0:
                records_per_second = result.successful_records / result.duration_seconds
                report_lines.append(f"Migration Speed: {records_per_second:,.0f} records/second")
        
        report_lines.append("")
        
        # Validation Results
        if validation_result:
            report_lines.append("[OK] VALIDATION RESULTS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Source Records: {validation_result['source_count']:,}")
            report_lines.append(f"Target Records: {validation_result['target_count']:,}")
            report_lines.append(f"Count Match: {'[OK] Yes' if validation_result['count_match'] else '[ERROR] No'}")
            
            if validation_result['sample_validation']:
                matching_samples = sum(1 for s in validation_result['sample_validation'] if s['match'])
                total_samples = len(validation_result['sample_validation'])
                report_lines.append(f"Sample Data Match: {matching_samples}/{total_samples}")
        
        # Error Summary
        if result.errors:
            report_lines.append("")
            report_lines.append("[ERROR] ERRORS ENCOUNTERED:")
            report_lines.append("-" * 40)
            for i, error in enumerate(result.errors[:10], 1):  # Show first 10 errors
                report_lines.append(f"{i}. {error}")
            
            if len(result.errors) > 10:
                report_lines.append(f"... and {len(result.errors) - 10} more errors")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # Save report to file with UTF-8 encoding
        report_filename = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[REPORT] Migration report saved to: {report_filename}")
        
        return report_content
    
    def close_connections(self):
        """Close database connections"""
        if self.source_connection:
            self.source_connection.close()
            logger.info("[OK] Source database connection closed")
        
        if self.target_connection:
            self.target_connection.close()
            logger.info("[OK] Target database connection closed")

def main_migration():
    """Main migration function"""
    
    print("[START] Starting Data Migration Process...")
    
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
    
    # Migration settings
    source_table = "Customers_AX"
    target_table = "BusinessPartner_SAP"
    batch_size = 1000
    
    # Initialize migration engine
    migrator = DataMigrationEngine(source_config, target_config)
    
    try:
        # Step 1: Connect to databases
        if not migrator.connect_databases():
            print("[ERROR] Failed to connect to databases")
            return
        
        # Step 2: Load column mappings
        if not migrator.load_mappings():
            print("[ERROR] Failed to load column mappings")
            return
        
        # Step 3: Validate mappings
        if not migrator.validate_mappings(source_table, target_table):
            print("[ERROR] Column mapping validation failed")
            return
        
        # Step 4: Migrate data
        print(f"\n[PROCESS] Starting migration from {source_table} to {target_table}...")
        
        # Choose migration method
        # Method 1: Batch migration (recommended for large datasets)
        migration_result = migrator.migrate_data_batch(
            source_table=source_table,
            target_table=target_table,
            batch_size=batch_size,
            truncate_target=True  # Set to False if you want to append data
        )
        
        # Method 2: Pandas migration (alternative)
        # migration_result = migrator.migrate_data_pandas(
        #     source_table=source_table,
        #     target_table=target_table,
        #     chunk_size=10000
        # )
        
        # Step 5: Validate migration
        print("\n[VALIDATE] Validating migration...")
        validation_result = migrator.validate_migration(source_table, target_table)
        
        # Step 6: Generate report
        print("\n[REPORT] Generating migration report...")
        report = migrator.generate_migration_report(migration_result, validation_result)
        print(report)
        
        # Summary
        if migration_result.successful_records > 0:
            print(f"\n[OK] Migration completed successfully!")
            print(f"   Migrated: {migration_result.successful_records:,} records")
            print(f"   Duration: {migration_result.duration_seconds:.2f} seconds")
        else:
            print(f"\n[ERROR] Migration failed!")
            print(f"   Errors: {len(migration_result.errors)}")
        
    except Exception as e:
        logger.error(f"[ERROR] Migration process failed: {str(e)}")
        print(f"[ERROR] Migration process failed: {str(e)}")
    
    finally:
        # Clean up connections
        migrator.close_connections()

if __name__ == "__main__":
    main_migration()