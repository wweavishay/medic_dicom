import os
import pydicom
import pyodbc


def read_dicom_files(folder_path, force=False):
    dicom_files = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.dcm'):
                file_path = os.path.join(dirpath, filename)
                try:
                    dicom_data = pydicom.dcmread(file_path, force=force)
                    dicom_files.append((filename, dicom_data))
                except pydicom.errors.InvalidDicomError as e:
                    print(f"Error reading DICOM file '{file_path}': {e}")
    return dicom_files



def create_connection(server_name, db_name):
    conn_str = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={db_name};Trusted_Connection=yes;'
    return pyodbc.connect(conn_str)


def create_table(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'DICOMMetadata'")
        if cursor.fetchone():
            print("Table 'DICOMMetadata' already exists in the database.")
            return False
        else:
            cursor.execute('''
                CREATE TABLE DICOMMetadata (
                    ID INT IDENTITY(1,1) PRIMARY KEY,
                    FileName NVARCHAR(255),
                    Date DATE,
                    TypeofDisease NVARCHAR(50),
                    StudyDescription NVARCHAR(255),
                    PatientID NVARCHAR(150),
                    ProtocolName NVARCHAR(255),
                    BodyPartExamined NVARCHAR(50)
                )
            ''')
            conn.commit()
            print("Table 'DICOMMetadata' created successfully.")
            return True
    except Exception as e:
        print("An error occurred while creating the table:")
        print(e)
        raise e

def create_table_study_info(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'StudyInformation'")
        if cursor.fetchone():
            print("Table 'StudyInformation' already exists in the database.")
            return False
        else:
            cursor.execute('''
                CREATE TABLE StudyInformation (
                    ID INT IDENTITY(1,1) PRIMARY KEY,
                    StudyInstanceUID NVARCHAR(255),
                    StudyDate NVARCHAR(255),
                    StudyTime NVARCHAR(255),
                    AccessionNumber NVARCHAR(50),
                    StudyDescription NVARCHAR(255)
                )
            ''')
            conn.commit()
            print("Table 'StudyInformation' created successfully.")
            return True
    except Exception as e:
        print("An error occurred while creating the table 'StudyInformation':")
        print(e)
        raise e


def create_table_series_info(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'SeriesInformation'")
        if cursor.fetchone():
            print("Table 'SeriesInformation' already exists in the database.")
            return False
        else:
            cursor.execute('''
                CREATE TABLE SeriesInformation (
                    ID INT IDENTITY(1,1) PRIMARY KEY,
                    SeriesInstanceUID NVARCHAR(255),
                    SeriesDescription NVARCHAR(255),
                    Modality NVARCHAR(50),
                    BodyPartExamined NVARCHAR(50),
                    SeriesNumber INT,
                    ImageCount INT
                )
            ''')
            conn.commit()
            print("Table 'SeriesInformation' created successfully.")
            return True
    except Exception as e:
        print("An error occurred while creating the table 'SeriesInformation':")
        print(e)
        raise e


def insert_records_study_info(conn, folder_name, dicom_files):
    cursor = conn.cursor()
    for _, dicom_data in dicom_files:
        study_instance_uid = dicom_data.StudyInstanceUID if hasattr(dicom_data, 'StudyInstanceUID') else None
        study_date = dicom_data.StudyDate if hasattr(dicom_data, 'StudyDate') else None
        study_time = dicom_data.StudyTime if hasattr(dicom_data, 'StudyTime') else None
        accession_number = dicom_data.AccessionNumber if hasattr(dicom_data, 'AccessionNumber') else None
        study_description = dicom_data.StudyDescription if hasattr(dicom_data, 'StudyDescription') else None

        cursor.execute('''
            INSERT INTO StudyInformation (StudyInstanceUID, StudyDate, StudyTime, AccessionNumber, StudyDescription)
            VALUES (?, ?, ?, ?, ?)
        ''', (study_instance_uid, study_date, study_time, accession_number, study_description))

    conn.commit()


def insert_records_series_info(conn, folder_name, dicom_files):
    cursor = conn.cursor()
    for _, dicom_data in dicom_files:
        series_instance_uid = dicom_data.SeriesInstanceUID if hasattr(dicom_data, 'SeriesInstanceUID') else None
        series_description = dicom_data.SeriesDescription if hasattr(dicom_data, 'SeriesDescription') else None
        modality = dicom_data.Modality if hasattr(dicom_data, 'Modality') else None
        body_part_examined = dicom_data.BodyPartExamined if hasattr(dicom_data, 'BodyPartExamined') else None
        series_number = dicom_data.SeriesNumber if hasattr(dicom_data, 'SeriesNumber') else None
        image_count = dicom_data.NumberOfImages if hasattr(dicom_data, 'NumberOfImages') else None

        cursor.execute('''
            INSERT INTO SeriesInformation (SeriesInstanceUID, SeriesDescription, Modality, BodyPartExamined, SeriesNumber, ImageCount)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (series_instance_uid, series_description, modality, body_part_examined, series_number, image_count))

    conn.commit()
def insert_records(conn, folder_name, dicom_files):
    cursor = conn.cursor()
    cursor.execute('SELECT FileName FROM DICOMMetadata')
    existing_filenames = [row[0] for row in cursor.fetchall()]

    for filename, dicom_data in dicom_files:
        if filename in existing_filenames:
            print(f"Row with filename '{filename}' already exists. Skipping insertion.")
            continue

        date = dicom_data.StudyDate if hasattr(dicom_data, 'StudyDate') else None
        study_description = dicom_data.StudyDescription if hasattr(dicom_data, 'StudyDescription') else None
        patient_id = dicom_data.PatientID if hasattr(dicom_data, 'PatientID') else "None"

        protocol_name = dicom_data.ProtocolName if hasattr(dicom_data, 'ProtocolName') else None
        body_part_examined = dicom_data.BodyPartExamined if hasattr(dicom_data, 'BodyPartExamined') else None

        cursor.execute('''
            INSERT INTO DICOMMetadata (FileName, Date, TypeofDisease, StudyDescription, PatientID, ProtocolName, BodyPartExamined)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (filename, date, folder_name, study_description, patient_id, protocol_name, body_part_examined))

    conn.commit()

def create_table_patient_info(conn):
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'PatientInformation'")
        if cursor.fetchone():
            print("Table 'PatientInformation' already exists in the database.")
            return False
        else:
            cursor.execute('''
                CREATE TABLE PatientInformation (
                    ID INT IDENTITY(1,1) PRIMARY KEY,
                    PatientID NVARCHAR(50),
                    PatientName NVARCHAR(255),
                    BirthDate DATE,
                    Sex NVARCHAR(10),
                    Age INT,
                    Weight FLOAT
                )
            ''')
            conn.commit()
            print("Table 'PatientInformation' created successfully.")
            return True
    except Exception as e:
        print("An error occurred while creating the table 'PatientInformation':")
        print(e)
        raise e

def insert_records_patient_info(conn, folder_name, dicom_files):
    cursor = conn.cursor()
    cursor.execute('SELECT PatientID FROM PatientInformation')
    existing_patient_ids = [row[0] for row in cursor.fetchall()]

    for _, dicom_data in dicom_files:
        patient_id = dicom_data.PatientID if hasattr(dicom_data, 'PatientID') else None

        # Check if patient ID already exists in the table
        if patient_id in existing_patient_ids:
            print(f"Patient with ID '{patient_id}' already exists. Skipping insertion.")
            continue

        # Handle patient name extraction more accurately
        patient_name = "Unknown"
        if hasattr(dicom_data, 'PatientName'):
            patient_name = dicom_data.PatientName.family_name if dicom_data.PatientName.family_name else ""
            patient_name += " " + dicom_data.PatientName.given_name if dicom_data.PatientName.given_name else ""

        birth_date = dicom_data.PatientBirthDate if hasattr(dicom_data, 'PatientBirthDate') else None
        sex = dicom_data.PatientSex if hasattr(dicom_data, 'PatientSex') else None

        # Extracting age from DICOM data may require additional processing
        patient_age_str = dicom_data.PatientAge if hasattr(dicom_data, 'PatientAge') else None
        patient_age = None
        if patient_age_str:
            try:
                patient_age = int(patient_age_str[:-1])  # Extract numeric part and remove suffix
            except ValueError:
                pass

        # Weight information may not be directly available in all DICOM files
        weight = None

        # Insert the record into the table
        cursor.execute('''
            INSERT INTO PatientInformation (PatientID, PatientName, BirthDate, Sex, Age, Weight)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (patient_id, patient_name, birth_date, sex, patient_age, weight))

        # Add the inserted patient ID to the list of existing patient IDs
        existing_patient_ids.append(patient_id)

    conn.commit()

def main():
    server_name = "DESKTOP-4VV2GQ3"
    db_name = "medicdb"
    folder_paths = ["A2", "A3" , "N1", "N2", "N3", "N5",
                    "N6", "PD1", "PD3", "PD4", "PD5", "PD6"]

    # Connect to SQL Server
    conn = create_connection(server_name, db_name)

    # Create tables if not exists
    create_table_patient_info(conn)

    # Similarly, create other tables
    for folder_path in folder_paths:
        dicom_files = read_dicom_files(folder_path, force=True)

        # Insert records into tables for each category of information
        # You need to implement similar insert functions for each table
        insert_records_patient_info(conn, folder_path, dicom_files)

    create_table_study_info(conn)
    create_table_series_info(conn)
    insert_records_study_info(conn, folder_path, dicom_files)
    insert_records_series_info(conn, folder_path, dicom_files)

    # Create table if not exists
    if create_table(conn):
        print("Table 'DICOMMetadata' was created.")

    for folder_path in folder_paths:
        dicom_files = read_dicom_files(folder_path)
        insert_records(conn, folder_path, dicom_files)
        print(f"DICOM files metadata for '{folder_path}' have been successfully added to the SQL Server database.")

    conn.close()
    print("All DICOM files metadata have been successfully added to the SQL Server database.")


if __name__ == "__main__":
    main()
