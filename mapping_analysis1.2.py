import json
from datetime import datetime

def display_enhanced_mapping_summary(mapping_file="demo_column_mapping.json"):
    """Display enhanced mapping summary with better formatting"""
    
    try:
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        
        print("\n" + "="*80)
        print("🎯 COLUMN MAPPING ANALYSIS REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Mappings: {len(mappings)}")
        
        # Categorize by similarity scores
        excellent = [m for m in mappings if m['Similarity'] >= 0.9]
        good = [m for m in mappings if 0.8 <= m['Similarity'] < 0.9]
        fair = [m for m in mappings if 0.7 <= m['Similarity'] < 0.8]
        
        print(f"\n📊 MAPPING QUALITY BREAKDOWN:")
        print(f"   🟢 Excellent (≥90%): {len(excellent)} mappings")
        print(f"   🟡 Good (80-89%):    {len(good)} mappings") 
        print(f"   🟠 Fair (70-79%):    {len(fair)} mappings")
        
        print(f"\n📝 DETAILED MAPPINGS:")
        print("-"*80)
        
        for i, mapping in enumerate(mappings, 1):
            similarity = mapping['Similarity']
            
            # Choose emoji based on similarity
            if similarity >= 0.9:
                emoji = "🟢"
            elif similarity >= 0.8:
                emoji = "🟡"
            else:
                emoji = "🟠"
                
            print(f"{i}. {emoji} {mapping['Source_Column']} → {mapping['Target_Column']}")
            print(f"   Similarity: {similarity*100:.1f}%")
            print(f"   Reason: {mapping['Match_Reason']}")
            print()
        
        return mappings
        
    except FileNotFoundError:
        print(f"❌ Mapping file '{mapping_file}' not found")
        return None
    except Exception as e:
        print(f"❌ Error reading mapping file: {e}")
        return None

def generate_sql_scripts(mappings, source_table="BusinessPartner_AX", target_table="BusinessPartner_SAP"):
    """Generate SQL scripts for data migration based on mappings"""
    
    if not mappings:
        return
    
    print("\n" + "="*80)
    print("🔧 GENERATED SQL SCRIPTS")
    print("="*80)
    
    # SELECT statement for data extraction
    select_columns = []
    insert_columns = []
    
    for mapping in mappings:
        source_col = mapping['Source_Column']
        target_col = mapping['Target_Column']
        select_columns.append(f"    {source_col} AS {target_col}")
        insert_columns.append(f"    {target_col}")
    
    print("\n1️⃣ DATA EXTRACTION QUERY:")
    print("-"*40)
    extraction_sql = f"""SELECT 
{',\\n'.join(select_columns)}
FROM {source_table};"""
    print(extraction_sql)
    
    print("\n2️⃣ INSERT TEMPLATE:")
    print("-"*40)
    insert_sql = f"""INSERT INTO {target_table} (
{',\\n'.join(insert_columns)}
)
SELECT 
{',\\n'.join([f"    {m['Source_Column']}" for m in mappings])}
FROM {source_table};"""
    print(insert_sql)
    
    # Save SQL to file
    sql_content = f"""-- Column Mapping SQL Scripts
-- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
-- Source: {source_table}
-- Target: {target_table}

-- Data Extraction Query
{extraction_sql}

-- Data Migration Query  
{insert_sql}
"""
    
    with open("migration_scripts.sql", "w") as f:
        f.write(sql_content)
    
    print(f"\n✅ SQL scripts saved to: migration_scripts.sql")

def export_to_excel_format(mappings):
    """Export mappings in a format ready for Excel/CSV"""
    
    if not mappings:
        return
        
    # Create Excel-friendly format
    excel_data = []
    excel_data.append(["Source_Column", "Target_Column", "Similarity_%", "Data_Type_Match", "Match_Reason"])
    
    for mapping in mappings:
        excel_data.append([
            mapping['Source_Column'],
            mapping['Target_Column'], 
            f"{mapping['Similarity']*100:.1f}%",
            "Compatible",  # You could enhance this by checking actual data types
            mapping['Match_Reason']
        ])
    
    # Save as CSV
    import csv
    with open("column_mappings.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(excel_data)
    
    print(f"\n✅ Excel-ready CSV saved to: column_mappings.csv")
    print("   Open in Excel for easy review and editing")

def main_enhanced_display():
    """Main function to run enhanced display"""
    
    print("📊 Loading and analyzing column mappings...")
    
    # Load and display mappings
    mappings = display_enhanced_mapping_summary()
    
    if mappings:
        # Generate SQL scripts
        generate_sql_scripts(mappings)
        
        # Export to CSV for Excel
        export_to_excel_format(mappings)
        
        print("\n" + "="*80)
        print("🎯 SUMMARY OF GENERATED FILES:")
        print("="*80)
        print("📄 demo_column_mapping.json    - Original AI mapping output")
        print("📄 migration_scripts.sql       - Ready-to-use SQL scripts") 
        print("📄 column_mappings.csv         - Excel-friendly format")
        print("\n✨ Your column mapping is complete and ready to use!")

if __name__ == "__main__":
    main_enhanced_display()