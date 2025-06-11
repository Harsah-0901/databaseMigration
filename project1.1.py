import google.generativeai as genai
import json
import os
import pyodbc
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Load API key from .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")  # using the free version

class DatabaseConfig:
    """Database connection configuration"""
    def __init__(self, server: str, database: str, username: str = None, password: str = None, trusted_connection: bool = True):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.trusted_connection = trusted_connection

class SchemaExtractor:
    """Extract schema information from SQL Server databases"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = None
    
    def connect(self) -> bool:
        """Connect to SQL Server"""
        try:
            if self.config.trusted_connection:
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.config.server};DATABASE={self.config.database};Trusted_Connection=yes;"
            else:
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.config.server};DATABASE={self.config.database};UID={self.config.username};PWD={self.config.password};"
            
            self.connection = pyodbc.connect(conn_str)
            print(f"Successfully connected to SQL Server: {self.config.server}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to SQL Server. Here's what went wrong: {str(e)}")
            return False
    
    def get_table_schema(self, table_name: str, schema_name: str = "dbo") -> Optional[str]:
        """Get schema information for a specific table"""
        if not self.connection:
            print("Oops, looks like there's no database connection established.")
            return None
        
        cursor = self.connection.cursor()
        
        try:
            # Get column information
            cursor.execute("""
                SELECT 
                    c.COLUMN_NAME,
                    c.DATA_TYPE,
                    c.CHARACTER_MAXIMUM_LENGTH,
                    c.NUMERIC_PRECISION,
                    c.NUMERIC_SCALE,
                    c.IS_NULLABLE,
                    c.COLUMN_DEFAULT,
                    CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 'YES' ELSE 'NO' END as IS_PRIMARY_KEY
                FROM INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN (
                    SELECT ku.TABLE_SCHEMA, ku.TABLE_NAME, ku.COLUMN_NAME
                    FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                    INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku 
                        ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                    WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                ) pk ON c.TABLE_SCHEMA = pk.TABLE_SCHEMA 
                    AND c.TABLE_NAME = pk.TABLE_NAME 
                    AND c.COLUMN_NAME = pk.COLUMN_NAME
                WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
                ORDER BY c.ORDINAL_POSITION
            """, (schema_name, table_name))
            
            rows = cursor.fetchall()
            
            if not rows:
                print(f"Heads up! Table '{schema_name}.{table_name}' wasn't found or doesn't have any columns.")
                # Let's list available tables to help debug
                print("Here are the available tables for your reference:")
                tables = self.get_all_tables(schema_name)
                for table in tables:
                    print(f"   - {table}")
                return None
            
            # Build schema string
            schema_text = f"Table: {table_name}\nColumns:\n"
            
            for row in rows:
                col_name = row[0]
                data_type = row[1].upper()
                max_length = row[2]
                precision = row[3]
                scale = row[4]
                is_nullable = row[5] == 'YES'
                default_val = row[6]
                is_pk = row[7] == 'YES'
                
                # Format data type with length/precision
                if data_type in ['VARCHAR', 'NVARCHAR', 'CHAR', 'NCHAR']:
                    if max_length == -1:
                        data_type_full = f"{data_type}(MAX)"
                    else:
                        data_type_full = f"{data_type}({max_length})"
                elif data_type in ['DECIMAL', 'NUMERIC']:
                    data_type_full = f"{data_type}({precision},{scale})"
                else:
                    data_type_full = data_type
                
                # Add column info
                pk_indicator = " [PRIMARY KEY]" if is_pk else ""
                nullable_indicator = " [NULL]" if is_nullable else " [NOT NULL]"
                default_indicator = f" [DEFAULT: {default_val}]" if default_val else ""
                
                schema_text += f"- {col_name}: {data_type_full}{pk_indicator}{nullable_indicator}{default_indicator}\n"
            
            return schema_text.strip()
            
        except Exception as e:
            print(f"Ran into an error while trying to get the schema for table {table_name}: {str(e)}")
            return None
        finally:
            cursor.close()
    
    def get_all_tables(self, schema_name: str = "dbo") -> List[str]:
        """Get list of all tables in the database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        
        try:
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = ?
                ORDER BY TABLE_NAME
            """, (schema_name,))
            
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            print(f"Trouble getting the table list: {str(e)}")
            return []
        finally:
            cursor.close()
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Database connection gracefully closed.")

def create_column_mapping(source_schema: str, target_schema: str, source_table: str, target_table: str, source_system: str = "Source", target_system: str = "Target") -> str:
    """Create column mapping using Gemini AI"""
    
    prompt = f"""
You are a database expert. Match columns from {source_system} table ({source_table}) to {target_system} table ({target_table}).

Analyze the column names, data types, and purposes to find the best matches.
Consider:
1. Exact name matches (highest similarity)
2. Similar names (e.g., CustID vs CustomerID)
3. Similar purposes (e.g., Email vs EmailAddress)
4. Data type compatibility

Output should be in JSON list format like:
[
    {{
        "Source_Column": "column_name",
        "Target_Column": "column_name", 
        "Similarity": 0.95,
        "Match_Reason": "explanation of why these columns match"
    }},
    ...
]

{source_system} Schema ({source_table}):
{source_schema}

{target_system} Schema ({target_table}):
{target_schema}
"""  
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred while generating the mapping with AI: {str(e)}")
        return None

def main():
    """Main function to demonstrate automatic schema extraction and mapping for multiple tables"""
    
    # Database configurations - FIXED: Using raw strings to handle backslashes
    source_db_config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS",   # Using raw string
        database="ax_db",   # your source database name
        trusted_connection=True   # set to False if using SQL authentication
    )
    
    target_db_config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS",   # Using raw string
        database="sap_db",   # your target database name
        trusted_connection=True   # set to False if using SQL authentication
    )
    
    # ============================================================================
    # 🔥 TABLE MAPPING CONFIGURATION 🔥
    # ============================================================================
    # Related table pairs - each source table maps to its corresponding target table
    # Position matters: source_tables[0] maps to target_tables[0], etc.
    
    source_tables = [
        "Customers_AX",      # Maps to BusinessPartner_SAP
        "Products_AX",       # Maps to MaterialMaster_SAP
        "Orders_AX",         # Maps to SalesOrders_SAP
        "Vendors_AX"         # Maps to Suppliermaster_SAP
    ]
    
    target_tables = [
        "BusinessPartner_SAP",    # Corresponds to Customers_AX
        "MaterialMaster_SAP",     # Corresponds to Products_AX
        "SalesOrders_SAP",        # Corresponds to Orders_AX
        "Suppliermaster_SAP"      # Corresponds to Vendors_AX
    ]
    # ============================================================================
    
    schema_name = "dbo"   # Schema is passed separately
    
    # Validate that source and target tables have the same count
    if len(source_tables) != len(target_tables):
        print(f"❌ ERROR: Number of source tables ({len(source_tables)}) must match number of target tables ({len(target_tables)})")
        print("Each source table should have a corresponding target table at the same position.")
        return
    
    print("Alright, let's get started with automatic schema extraction and mapping for related table pairs...")
    print(f"Table pairs to process: {len(source_tables)}")
    print("Related table mappings:")
    for i, (src, tgt) in enumerate(zip(source_tables, target_tables)):
        print(f"   {i+1}. {src} → {tgt}")
    
    # Initialize extractors
    source_extractor = SchemaExtractor(source_db_config)
    target_extractor = SchemaExtractor(target_db_config)
    
    # Connect to both databases
    print(f"\nConnecting to source database: {source_db_config.database}")
    if not source_extractor.connect():
        print("Couldn't connect to the source database. Please check your connection details.")
        return
    
    print(f"Connecting to target database: {target_db_config.database}")
    if not target_extractor.connect():
        print("Couldn't connect to the target database. Please check your connection details.")
        source_extractor.disconnect()
        return
    
    successful_mappings = 0
    failed_mappings = 0
    mapping_summary = []
    
    try:
        # Main processing loop - iterate through related table pairs only
        for i, (source_table, target_table) in enumerate(zip(source_tables, target_tables)):
            print(f"\n" + "="*60)
            print(f"Processing Pair {i+1}/{len(source_tables)}: {source_table} → {target_table}")
            print("="*60)
            
            # Extract source schema
            print(f"📥 Extracting schema from: {schema_name}.{source_table}")
            source_schema = source_extractor.get_table_schema(source_table, schema_name)
            
            if not source_schema:
                print(f"❌ Failed to extract schema for source table: {source_table}")
                failed_mappings += 1
                continue
            
            print(f"✅ Source schema extracted successfully for {source_table}")
            
            # Extract target schema
            print(f"📥 Extracting schema from: {schema_name}.{target_table}")
            target_schema = target_extractor.get_table_schema(target_table, schema_name)
            
            if not target_schema:
                print(f"❌ Failed to extract schema for target table: {target_table}")
                failed_mappings += 1
                continue
            
            print(f"✅ Target schema extracted successfully for {target_table}")
            
            # Generate mapping using AI
            print("🤖 AI is generating column mappings...")
            mapping_response = create_column_mapping(
                source_schema, 
                target_schema, 
                source_table,
                target_table,
                "AX", 
                "SAP"
            )
            
            if mapping_response:
                try:
                    # Clean up the response (remove markdown formatting if present)
                    clean_response = mapping_response.strip()
                    if clean_response.startswith("```json"):
                        clean_response = clean_response[7:]
                    if clean_response.endswith("```"):
                        clean_response = clean_response[:-3]
                    clean_response = clean_response.strip()
                    
                    mapping = json.loads(clean_response)
                    
                    # Create unique filename for this mapping
                    output_file = f"mapping_{source_table}_to_{target_table}.json"
                    
                    # Add metadata to the mapping
                    mapping_with_metadata = {
                        "source_table": source_table,
                        "target_table": target_table,
                        "source_database": source_db_config.database,
                        "target_database": target_db_config.database,
                        "generation_timestamp": "auto-generated",
                        "total_mappings": len(mapping),
                        "mappings": mapping
                    }
                    
                    # Save to file
                    with open(output_file, "w") as f:
                        json.dump(mapping_with_metadata, f, indent=4)
                    
                    print(f"✅ Mapping saved to: {output_file}")
                    
                    # Track summary
                    high_similarity = [m for m in mapping if m.get('Similarity', 0) >= 0.9]
                    mapping_summary.append({
                        "source": source_table,
                        "target": target_table,
                        "total_mappings": len(mapping),
                        "high_confidence": len(high_similarity),
                        "filename": output_file
                    })
                    
                    successful_mappings += 1
                    print(f"📊 Found {len(mapping)} column mappings ({len(high_similarity)} high confidence)")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Couldn't parse AI response as JSON for {source_table} → {target_table}")
                    
                    # Save raw response for manual review
                    raw_file = f"raw_mapping_{source_table}_to_{target_table}.txt"
                    with open(raw_file, "w") as f:
                        f.write(f"Source: {source_table}\nTarget: {target_table}\n\n")
                        f.write(mapping_response)
                    print(f"Raw response saved to: {raw_file}")
                    failed_mappings += 1
                    
                except Exception as e:
                    print(f"❌ Unexpected error processing {source_table} → {target_table}: {str(e)}")
                    failed_mappings += 1
            else:
                print(f"❌ Couldn't generate mapping for {source_table} → {target_table}")
                failed_mappings += 1
    
    finally:
        # Clean up connections
        source_extractor.disconnect()
        target_extractor.disconnect()
    
    print(f"\n" + "="*60)
    print("🎉 PROCESSING COMPLETE - FINAL SUMMARY")
    print("="*60)
    print(f"✅ Successful mappings: {successful_mappings}")
    print(f"❌ Failed mappings: {failed_mappings}")
    print(f"📊 Total table pairs processed: {successful_mappings + failed_mappings}")
    
    if mapping_summary:
        print(f"\n📋 DETAILED MAPPING SUMMARY:")
        for summary in mapping_summary:
            print(f"   ✅ {summary['source']} → {summary['target']}: "
                  f"{summary['total_mappings']} mappings "
                  f"({summary['high_confidence']} high confidence) "
                  f"→ {summary['filename']}")
        
        # Save summary to file
        with open("mapping_summary.json", "w") as f:
            json.dump({
                "total_successful": successful_mappings,
                "total_failed": failed_mappings,
                "table_pairs_processed": len(source_tables),
                "detailed_summary": mapping_summary
            }, f, indent=4)
        print(f"\n📄 Complete summary saved to: mapping_summary.json")
    
    print(f"\n🎯 All mapping files have been generated in the current directory!")
    print("📂 Generated Files:")
    for summary in mapping_summary:
        print(f"   - {summary['filename']}")
    print(f"   - mapping_summary.json")

def demo_with_sample_data():
    """Demo function using the original hardcoded schemas for testing"""
    print("Alright, running a quick demo with some sample data...")
    
    # Original sample schemas
    ax_schema = """Table: BusinessPartner_AX
Columns:
- CustID: INT [PRIMARY KEY] [NOT NULL]
- CustName: NVARCHAR(100) [NOT NULL]
- Email: NVARCHAR(100) [NULL]
- Phone: NVARCHAR(15) [NULL]
- Address: NVARCHAR(200) [NULL]"""

    sap_schema = """Table: BusinessPartner_SAP
Columns:
- BPID: INT [PRIMARY KEY] [NOT NULL]
- BPName: NVARCHAR(100) [NOT NULL]
- EmailAddress: NVARCHAR(100) [NULL]
- ContactNumber: NVARCHAR(15) [NULL]
- Location: NVARCHAR(200) [NULL]"""
    
    print("Using these sample schemas:")
    print(f"AX Schema:\n{ax_schema}\n")
    print(f"SAP Schema:\n{sap_schema}\n")
    
    # Generate mapping
    print("Having the AI generate column mappings for these samples...")
    mapping_response = create_column_mapping(ax_schema, sap_schema, "BusinessPartner_AX", "BusinessPartner_SAP", "AX", "SAP")
    
    if mapping_response:
        try:
            # Clean up the response
            clean_response = mapping_response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            mapping = json.loads(clean_response)
            print("Successfully parsed the demo mapping:\n", json.dumps(mapping, indent=4))
            
            with open("demo_column_mapping.json", "w") as f:
                json.dump(mapping, f, indent=4)
            print("The demo mapping has been saved to demo_column_mapping.json for you.")
            
        except Exception as e:
            print("Couldn't quite parse the response as JSON for the demo.")
            print("Here's the Raw AI Response:")
            print(mapping_response)

def debug_tables():
    """Debug function to list all available tables"""
    print("Just doing a quick check: Listing all available tables...")
    
    config = DatabaseConfig(
        server=r"localhost\SQLEXPRESS",
        database="ax_db",
        trusted_connection=True
    )
    
    extractor = SchemaExtractor(config)
    if extractor.connect():
        tables = extractor.get_all_tables("dbo")
        print(f"Found {len(tables)} tables in ax_db:")
        for table in tables:
            print(f"   - {table}")
        extractor.disconnect()
    else:
        print("Couldn't connect to the database for debugging purposes.")

if __name__ == "__main__":
    # First, let's see what tables are available
    debug_tables()
    
    print("\n" + "="*50 + "\n")
    
    # Choose which version to run:
    
    # Option 1: Run with actual database connections (update the configs above)
    main()
    
    # Option 2: Run demo with sample data (no database required)
    #demo_with_sample_data()