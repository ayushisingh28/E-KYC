import logging
import os

try:
    import mysql.connector
except Exception as exc:  # pragma: no cover - import guard for deployment environments
    mysql = None
    MYSQL_IMPORT_ERROR = exc
else:
    MYSQL_IMPORT_ERROR = None

import pandas as pd
import streamlit as st


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "ekyc_logs.log"),
    level=logging.INFO,
    format=logging_str,
    filemode="a",
)


def _get_db_config():
    """Load database credentials from environment variables and Streamlit secrets."""
    db_config = {}

    candidate_names = {
        "host": ["host", "MYSQL_HOST", "MYSQLHOST", "DB_HOST", "DATABASE_HOST"],
        "port": ["port", "MYSQL_PORT", "MYSQLPORT", "DB_PORT", "DATABASE_PORT"],
        "user": ["user", "MYSQL_USER", "MYSQLUSER", "DB_USER", "DATABASE_USER"],
        "password": ["password", "MYSQL_PASSWORD", "MYSQLPASSWORD", "DB_PASSWORD", "DATABASE_PASSWORD"],
        "database": ["database", "MYSQL_DATABASE", "MYSQLDATABASE", "DB_NAME", "DATABASE_NAME"],
    }

    for key, names in candidate_names.items():
        for name in names:
            value = os.getenv(name)
            if value not in (None, ""):
                db_config[key] = value
                break

    try:
        secret_config = st.secrets.get("database")
        if isinstance(secret_config, dict):
            for key, names in candidate_names.items():
                for name in names:
                    value = secret_config.get(name)
                    if value not in (None, ""):
                        db_config[key] = value
                        break
    except Exception:
        pass

    return db_config


def get_db_connection():
    """Open a MySQL connection using credentials from secrets or environment variables."""
    if mysql is None:
        message = f"MySQL connector is unavailable: {MYSQL_IMPORT_ERROR}"
        logging.error(message)
        raise RuntimeError(message)

    db_config = _get_db_config()
    if not db_config:
        message = "No MySQL configuration found. Add the [database] section in Streamlit secrets or set MYSQL_HOST/MYSQL_USER/MYSQL_PASSWORD/MYSQL_DATABASE."
        logging.error(message)
        raise RuntimeError(message)

    required_fields = ["host", "user", "password", "database"]
    missing_fields = [field for field in required_fields if not str(db_config.get(field, "")).strip()]
    if missing_fields:
        message = f"Missing MySQL config values: {', '.join(missing_fields)}"
        logging.error(message)
        raise RuntimeError(message)

    try:
        port = int(db_config.get("port", 3306))
    except (TypeError, ValueError) as error:
        message = f"Invalid MySQL port: {db_config.get('port')}"
        logging.error(message)
        raise RuntimeError(message) from error

    connection_kwargs = {
        "host": db_config["host"],
        "port": port,
        "user": db_config["user"],
        "password": db_config["password"],
        "database": db_config["database"],
        "connection_timeout": 30,
        "use_pure": True,
    }

    last_error = None
    for attempt in [
        connection_kwargs,
        {**connection_kwargs, "ssl_disabled": True},
        {**connection_kwargs, "ssl_disabled": False, "auth_plugin": "mysql_native_password"},
    ]:
        try:
            return mysql.connector.connect(**attempt)
        except Exception as error:
            last_error = error
            logging.warning("MySQL connection attempt failed: %s | kwargs=%s", error, attempt)

    message = f"Unable to connect to MySQL at '{db_config.get('host', '')}'. Last error: {last_error}"
    logging.error(message)
    raise RuntimeError(message) from last_error


def ensure_tables_exist(connection):
    """Create required tables if they do not already exist."""
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id VARCHAR(128) PRIMARY KEY,
                name VARCHAR(255),
                father_name VARCHAR(255),
                dob DATE,
                id_type VARCHAR(32),
                embedding LONGTEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS aadhar (
                id VARCHAR(128) PRIMARY KEY,
                name VARCHAR(255),
                gender VARCHAR(32),
                dob DATE,
                id_type VARCHAR(32),
                embedding LONGTEXT
            )
            """
        )
        connection.commit()
    finally:
        cursor.close()


def _open_database():
    connection = get_db_connection()
    ensure_tables_exist(connection)
    return connection


def insert_records(text_info):
    connection = None
    cursor = None
    try:
        connection = _open_database()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO users(id, name, father_name, dob, id_type, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                text_info["ID"], text_info["Name"], text_info.get("Father's Name", ""),
                text_info["DOB"], text_info.get("ID Type", "PAN"), str(text_info["Embedding"]),
            ),
        )
        connection.commit()
        return True, "Record inserted successfully."
    except Exception as error:
        logging.error("Error inserting PAN record: %s", error)
        return False, f"Could not insert record: {error}"
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def insert_records_aadhar(text_info):
    connection = None
    cursor = None
    try:
        connection = _open_database()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO aadhar(id, name, gender, dob, id_type, embedding) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                text_info["ID"], text_info["Name"], text_info.get("Gender", ""),
                text_info["DOB"], text_info.get("ID Type", "AADHAR"), str(text_info["Embedding"]),
            ),
        )
        connection.commit()
        return True, "Record inserted successfully."
    except Exception as error:
        logging.error("Error inserting Aadhaar record: %s", error)
        return False, f"Could not insert record: {error}"
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def _fetch_records(table_name, user_id):
    connection = None
    cursor = None
    try:
        connection = _open_database()
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE id = %s", (user_id,))
        records = cursor.fetchall()
        return pd.DataFrame(records, columns=[column[0] for column in cursor.description]) if records else pd.DataFrame()
    except Exception as error:
        logging.error("Error fetching %s records: %s", table_name, error)
        return pd.DataFrame()
    finally:
        if cursor:
            cursor.close()
        if connection and connection.is_connected():
            connection.close()


def fetch_records(text_info):
    return _fetch_records("users", text_info["ID"])


def fetch_records_aadhar(text_info):
    return _fetch_records("aadhar", text_info["ID"])


def check_duplicacy(text_info):
    return not fetch_records(text_info).empty


def check_duplicacy_aadhar(text_info):
    return not fetch_records_aadhar(text_info).empty
