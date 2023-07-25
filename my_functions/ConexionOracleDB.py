import cx_Oracle
import oracledb


class ConexionOracle:
    user = 'parqueo'
    password = 'parqueo2k23'
    host = '172.16.1.39'
    port = '1521'
    service_name = 'xepdb1'
    conn = None

    @classmethod
    def conectar(cls):
        params = oracledb.ConnectParams(host=cls.host, port=cls.port, service_name=cls.service_name)
        try:
            conn = oracledb.connect(user=cls.user, password=cls.password, params=params)
            print(conn)
            cls.conn = conn

        except oracledb.DatabaseError as e:
            print("Error while connecting to database", e)

    @classmethod
    def cerrar_conexion(cls):
        try:
            if cls.conn:
                cls.conn.close()
                cls.conn = None
                print("Conexion cerrada")
        except Exception as e:
            print("Error: ", e)

