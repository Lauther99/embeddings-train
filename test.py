import pyodbc
import pandas as pd
db_user: str = "lauther"
db_pwd: str = "123"
db_host: str = "192.168.1.122,1433"
db_name: str = "FMS_MMS_NTS_RED_FC_220525"
db_driver: str = "ODBC Driver 18 for SQL Server"

conn: pyodbc.Connection = pyodbc.connect(
    f"DRIVER={db_driver};SERVER={db_host};DATABASE={db_name};UID={db_user};PWD={db_pwd}"
)

sql = "SELECT * FROM dbo_v2.med_sistema_medicion"

df = pd.read_sql_query(sql, conn)

print(df.head(30))

