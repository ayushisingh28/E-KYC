import os
import logging
import streamlit as st
from preprocess import read_image, extract_id_card, save_image
from ocr_engine import extract_text
from postprocess import extract_information,extract_information1
from face_verification import detect_and_extract_face, deepface_face_comparison, get_face_embeddings
from sql_connection import insert_records, fetch_records, check_duplicacy,insert_records_aadhar,fetch_records_aadhar,check_duplicacy_aadhar
import toml
import hashlib
from urllib.parse import quote_plus
from datetime import datetime
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

config = toml.load("config.toml")
db_config = config.get("database", {})

db_user = db_config.get("user")
db_password = db_config.get("password")
db_host = db_config.get("host", "localhost")
db_name = db_config.get("database")

def hash_id(id_value):
    hash_object = hashlib.sha256(id_value.encode())
    hashed_id = hash_object.hexdigest()
    return hashed_id

def normalize_and_validate_text_info(text_info, option):
    required_fields = ["ID", "Name", "DOB"]
    if option != "PAN":
        required_fields.append("Gender")

    missing_fields = [
        field for field in required_fields
        if not str(text_info.get(field, "")).strip()
    ]
    if missing_fields:
        return False, f"Could not extract required field(s): {', '.join(missing_fields)}.", text_info

    dob = text_info.get("DOB")
    if isinstance(dob, datetime):
        text_info["DOB"] = dob.strftime("%Y-%m-%d")
    elif isinstance(dob, str):
        dob = dob.strip()
        parsed_dob = None
        for date_format in ("%d/%m/%Y", "%Y-%m-%d"):
            try:
                parsed_dob = datetime.strptime(dob, date_format)
                break
            except ValueError:
                continue
        if parsed_dob is None:
            return False, "Could not read DOB as a valid date.", text_info
        text_info["DOB"] = parsed_dob.strftime("%Y-%m-%d")
    else:
        return False, "Could not read DOB as a valid date.", text_info

    text_info["ID"] = str(text_info["ID"]).strip()
    text_info["Name"] = str(text_info["Name"]).strip()
    return True, "", text_info


def display_record_table(record):
    display_record = {
        key: value for key, value in record.items()
        if key != "Embedding"
    }
    st.dataframe([display_record], use_container_width=True)


def display_existing_records(records):
    display_records = records.drop(columns=["embedding"], errors="ignore")
    st.dataframe(display_records, use_container_width=True)


# Set wider page layout
def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Page layout set to wider configuration.")

# Customized Streamlit theme
def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6; /* Set background color */
                color: #333333; /* Set text color */
            }
            .sidebar .sidebar-content {
                background-color: #ffffff; /* Set sidebar background color */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Custom theme applied to Streamlit app.")


# Sidebar
def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    option = st.sidebar.selectbox(
        "ID Card Type",
        ("PAN", "AADHAR"),
        label_visibility="collapsed",
    )
    logging.info(f"ID card type selected: {option}")
    return option

# Header
def header_section(option):
    if option == "AADHAR":
        st.title("Registration Using Aadhar Card")
        logging.info("Header set for Aadhar Card registration.")
    elif option == "PAN":
        st.title("Registration Using PAN Card")
        logging.info("Header set for PAN Card registration.")


# Main content
def main_content(image_file, face_image_file,option):
    if image_file is not None:
        if face_image_file is None:
            st.error("Please upload a face image.")
            logging.error("No face image uploaded.")
            return

        face_image = read_image(face_image_file, is_uploaded=True)
        logging.info("Face image loaded.")
        if face_image is not None:
            image = read_image(image_file, is_uploaded=True)
            if image is None:
                st.error("Could not read the ID card image. Please upload a clear JPG/PNG image.")
                logging.error("ID card image could not be read.")
                return

            logging.info("ID card image loaded.")
            image_roi, _ = extract_id_card(image)
            if image_roi is None:
                st.error("Could not detect the ID card area. Please upload a clearer, uncropped card image.")
                logging.error("ID card ROI extraction failed.")
                return

            logging.info("ID card ROI extracted.")
            face_image_path2 = detect_and_extract_face(img=image_roi)
            if face_image_path2 is None:
                st.error("Could not detect a face on the ID card. Please upload a clearer card image.")
                logging.error("Face extraction from ID card failed.")
                return

            face_image_path1 = save_image(face_image, "face_image.jpg", path="data\\02_intermediate_data")
            logging.info("Faces extracted and saved.")
            is_face_verified, verification_message = deepface_face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
            logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

            if is_face_verified:
                extracted_text = extract_text(image_roi)
                text_info = extract_information(extracted_text) if option == "PAN" else extract_information1(extracted_text)
                # print(extracted_text)
                logging.info("Text extracted and information parsed from ID card.")
                is_valid_text_info, validation_message, text_info = normalize_and_validate_text_info(text_info, option)
                if not is_valid_text_info:
                    st.error(f"OCR extraction failed: {validation_message} Please upload a clearer ID card image.")
                    st.write(text_info)
                    logging.error(f"OCR extraction failed: {validation_message} Parsed data: {text_info}")
                    return

                display_text_info = text_info.copy()
                text_info['ID']=hash_id(text_info['ID'])
                records = fetch_records(text_info) if option=="PAN" else fetch_records_aadhar(text_info)
                if records.shape[0] > 0:
                    st.info("User already present.")
                    display_existing_records(records)
                    return
                is_duplicate = check_duplicacy(text_info) if option=="PAN" else check_duplicacy_aadhar(text_info)
                if is_duplicate:
                    st.info("User already present.")
                else: 
                    # text_info["ID"]=hash_id(text_info["ID"])
                    text_info['Embedding'] =  get_face_embeddings(face_image_path1)
                    is_inserted, insert_message = insert_records(text_info) if option == "PAN" else insert_records_aadhar(text_info)
                    if not is_inserted:
                        st.error(insert_message)
                        return
                    st.success("New user registered successfully.")
                    display_record_table(display_text_info)
                    logging.info(f"New user record inserted: {text_info['ID']}")
                    
            else:
                st.error(f"Face verification failed: {verification_message}")

        else:
            st.error("Face image not uploaded. Please upload a face image.")
            logging.error("No face image uploaded.")

    else:
        st.warning("Please upload an ID card image.")
        logging.warning("No ID card image uploaded.")

# Main function setup as previously provided...
def main():
    # Initialize connection.
    db_url = f"mysql://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:3306/{db_name}"
    conn = st.connection(
        "local_db",
        type="sql",
        url=db_url,
    )
    wider_page()
    set_custom_theme()
    option = sidebar_section()
    header_section(option)
    image_file = st.file_uploader("Upload ID Card")
    if image_file is not None:
        face_image_file = st.file_uploader("Upload Face Image")
        main_content(image_file, face_image_file,option)

if __name__ == "__main__":
    main()
