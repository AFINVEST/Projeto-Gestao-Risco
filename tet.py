import psycopg2

table_name = "AF DEB INCENTIVAS"

# Conexão corrigida
try:
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="hF81aEQwSDNzATLJ",
        host="db.obgwfekirteetqzjydry.supabase.co",
        port="5432"
    )
except Exception as e:
    print("Erro ao conectar ao banco de dados:", e)
    exit(1)
cursor = conn.cursor()

# Obter colunas existentes
cursor.execute(f"""
    SELECT column_name FROM information_schema.columns 
    WHERE table_name = '{table_name}';
""")
existing_columns = {row[0] for row in cursor.fetchall()}
print("Colunas existentes:", existing_columns)