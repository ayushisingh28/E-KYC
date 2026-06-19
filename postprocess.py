import pandas as pd
from datetime import datetime
import re


PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
DATE_PATTERNS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d%m%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d%m%y",
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%Y.%m.%d",
    "%Y%m%d",
    "%Y/%d/%m",
    "%Y-%d-%m",
    "%Y.%d.%m",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m.%d.%Y",
    "%m%d%Y",
    "%Y",
    "%y",
)


def _split_ocr_tokens(data_string):
    cleaned = data_string.replace(".", "/")
    return [word.strip() for word in cleaned.split("|") if word.strip()]


def _normalise_pan_candidate(value):
    return (
        value.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("/", "")
    )


def _normalise_pan_window(value):
    letter_fixes = {
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B",
    }
    digit_fixes = {
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "Z": "7",
        "B": "8",
    }
    chars = list(value.upper())
    for index in range(min(5, len(chars))):
        chars[index] = letter_fixes.get(chars[index], chars[index])
    for index in range(5, min(9, len(chars))):
        chars[index] = digit_fixes.get(chars[index], chars[index])
    if len(chars) == 10:
        chars[9] = letter_fixes.get(chars[9], chars[9])
    return "".join(chars)


def _parse_date(value):
    candidates = [value.strip()]
    digits = re.sub(r"\D", "", value)
    if len(digits) == 8:
        candidates.append(digits)
    candidates.extend(re.findall(r"\d{1,4}[\/\-.\s]\d{1,2}[\/\-.\s]\d{1,4}", value))
    if len(digits) in {2, 4}:
        candidates.append(digits)

    for candidate in candidates:
        for date_format in DATE_PATTERNS:
            try:
                parsed = datetime.strptime(candidate, date_format)
                if parsed.year < 100:
                    current_year = datetime.now().year % 100
                    if parsed.year <= current_year:
                        parsed = parsed.replace(year=parsed.year + 2000)
                    else:
                        parsed = parsed.replace(year=parsed.year + 1900)
                if parsed.year < 1900:
                    continue
                return parsed
            except ValueError:
                continue
    return None


def _is_label_or_noise(value):
    compact = re.sub(r"[^A-Z]", "", value.upper())
    if not compact:
        return True

    ignored_labels = {
        "PAN",
        "HR",
        "HRT",
        "GOVT",
        "GOVTOFINDIA",
        "GOVERNMENTOFINDIA",
        "GOVERNMENT",
        "COVERNNENT",
        "COVERNMENT",
        "INDIA",
        "INCOMETAXDEPARTMENT",
        "LNCOMETAXDEPARTMENT",
        "PERMANENTACCOUNTNUMBER",
        "PERMANENTACCOUNTNUMBERCARD",
        "NAME",
        "FATHERSNAME",
        "FATHERNAME",
        "DATEOFBIRTH",
        "DELEOFBIRTN",
        "SIGNATURE",
    }
    if compact in ignored_labels:
        return True
    if "GOV" in compact or "COVERN" in compact or "INDIA" in compact:
        return True
    return False


def _is_noise_token(value):
    compact = re.sub(r"[^A-Z]", "", value.upper())
    noise_fragments = {
        "GOV", "INDIA", "UIDAI", "AADHAAR", "AADHAR", "GOVERN", "COVERNNENT",
        "DEPARTMENT", "TAX", "GOVT", "QR", "CODE", "VID", "UID",
        "CARD", "GOVERNMENTOFINDIA", "INCOME", "TAXDEPARTMENT"
    }
    if not compact:
        return True
    for fragment in noise_fragments:
        if fragment in compact:
            return True
    return False


def _looks_like_person_name(value):
    if re.search(r"[^A-Za-z\s'.]", value):
        return False
    if _is_label_or_noise(value) or _is_noise_token(value):
        return False
    if PAN_PATTERN.search(_normalise_pan_candidate(value)):
        return False
    if _parse_date(value) is not None:
        return False
    letters = re.sub(r"[^A-Z ]", "", value.upper()).strip()
    return len(letters) >= 3 and not any(char.isdigit() for char in value)


def _clean_person_name(value):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z ]", " ", value.upper())).strip()


def _first_date(tokens):
    for index, token in enumerate(tokens[:-1]):
        compact = re.sub(r"[^A-Z]", "", token.upper())
        if "DOB" in compact or "BIRTH" in compact or "BIRTN" in compact:
            for next_token in tokens[index + 1:]:
                parsed_date = _parse_date(next_token)
                if parsed_date is not None:
                    return parsed_date

    for token in tokens:
        parsed_date = _parse_date(token)
        if parsed_date is not None:
            return parsed_date
    return ""


def _extract_pan_id(tokens, data_string):
    for candidate_text in tokens:
        compact = re.sub(r"[^A-Z0-9]", "", candidate_text.upper())
        for start in range(0, max(len(compact) - 9, 0)):
            candidate = _normalise_pan_window(compact[start:start + 10])
            if PAN_PATTERN.fullmatch(candidate):
                return candidate
    return ""


def _token_has_pan_id(value, pan_id):
    if not pan_id:
        return False
    compact = re.sub(r"[^A-Z0-9]", "", value.upper())
    for start in range(0, max(len(compact) - 9, 0)):
        if _normalise_pan_window(compact[start:start + 10]) == pan_id:
            return True
    return False


def _token_after_label(tokens, label_options):
    normalised_labels = {re.sub(r"[^A-Z]", "", label.upper()) for label in label_options}
    for index, token in enumerate(tokens[:-1]):
        compact = re.sub(r"[^A-Z]", "", token.upper())
        if compact in normalised_labels:
            for next_token in tokens[index + 1:]:
                if _looks_like_person_name(next_token):
                    return next_token.strip()
    return ""


def filter_lines(lines):
    start_index = None
    end_index = None

    # Find start and end indices
    start_index = None
    end_index = None

    for i in range(len(lines)):
       line = lines[i]
       if "INCOME TAX DEPARTMENT" in line and start_index is None:
           start_index = i
       if "Signature" in line:
          end_index = i
          break


    # Filter lines based on conditions
    filtered_lines = []
    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index + 1]:
            if len(line.strip()) > 2:
                filtered_lines.append(line.strip())
    
    return filtered_lines


# -------------- DEBUGGING ----------------

# Example list of lines
# lines = [
#     "Some irrelevant text",
#     "INCOME TAX DEPARTMENT",
#     "Line with relevant information",
#     "Signature",
#     "More irrelevant text"
# ]

# # Filter lines
# filtered_lines = filter_lines(lines)

# # Print filtered lines
# for line in filtered_lines:
#     print(line)


def create_dataframe(texts):

    lines = filter_lines(texts)
    print("="*20)
    print(lines)
    print("="*20)
    data = []
    name = lines[2].strip()
    father_name = lines[3].strip()
    dob = lines[4].strip()
    for i in range(len(lines)):
        if "Permanent Account Number" in lines[i]:
            pan = lines[i+1].strip()
    data.append({"ID": pan, "Name": name, "Father's Name": father_name, "DOB": dob, "ID Type": "PAN"})
    df = pd.DataFrame(data)
    return df

#-----------DEBUGGING------------------

# text=['8', '8', '3', 'HRT', 'INCOME TAX DEPARTMENT', 'GOVT OF INDIA', 'SUMIT', 'RAM SWARUP', '04/03/1992', 'Permanent Account Number', 'J', 'FZKPS9811P', 'Signature', '1', '2', '8']
# df=create_dataframe(text)
# print(df)


def extract_information(data_string):
    extracted_info = {
        "ID": "",
        "Name": "",
        "Father's Name": "",
        "DOB": "",
        "ID Type": "PAN"
    }

    words = [word for word in _split_ocr_tokens(data_string) if len(word.strip()) > 1]
    extracted_info["ID"] = _extract_pan_id(words, data_string)
    extracted_info["DOB"] = _first_date(words)

    labelled_name = _token_after_label(words, {"Name"})
    labelled_father = _token_after_label(words, {"Father's Name", "Father Name", "Fathers Name"})

    name_candidates = [word.strip() for word in words if _looks_like_person_name(word)]
    if labelled_name:
        extracted_info["Name"] = labelled_name
    elif extracted_info["ID"]:
        id_index = next(
            (
                index for index, word in enumerate(words)
                if _token_has_pan_id(word, extracted_info["ID"])
            ),
            -1,
        )
        candidates_after_id = [
            word.strip()
            for word in words[id_index + 1:]
            if _looks_like_person_name(word)
        ]
        if candidates_after_id:
            extracted_info["Name"] = candidates_after_id[0]
    elif name_candidates:
        extracted_info["Name"] = name_candidates[0]

    if labelled_father:
        extracted_info["Father's Name"] = labelled_father
    else:
        name_index = next(
            (
                index for index, word in enumerate(words)
                if word.strip().upper() == extracted_info["Name"].upper()
            ),
            -1,
        )
        remaining_names = [
            name for name in words[name_index + 1:]
            if _looks_like_person_name(name)
            and name.upper() != extracted_info["Name"].upper()
        ]
        if remaining_names:
            extracted_info["Father's Name"] = remaining_names[0]

    extracted_info["Name"] = _clean_person_name(extracted_info["Name"])
    extracted_info["Father's Name"] = _clean_person_name(extracted_info["Father's Name"])
    return extracted_info


def _split_text_tokens(data_string):
    tokens = re.split(r"[|\n\r]+", data_string)
    return [token.strip() for token in tokens if token.strip()]


def _normalize_aadhar_candidate(value):
    return (
        value.upper()
        .replace(" ", "")
        .replace("-", "")
        .replace("O", "0")
        .replace("Q", "0")
        .replace("D", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("Z", "2")
        .replace("S", "5")
        .replace("B", "8")
        .replace("G", "6")
    )


def _find_aadhar_id(tokens, data_string):
    joined = " ".join(tokens)
    match = re.search(r"\b(\d{4}\s*\d{4}\s*\d{4})\b", joined)
    if match:
        digits = re.sub(r"\D", "", match.group(1))
        return " ".join([digits[i:i+4] for i in range(0, 12, 4)])

    match = re.search(r"\b(\d{12})\b", joined)
    if match:
        digits = match.group(1)
        return " ".join([digits[i:i+4] for i in range(0, 12, 4)])

    normalized_joined = _normalize_aadhar_candidate(joined)
    match = re.search(r"\b(\d{12})\b", normalized_joined)
    if match:
        digits = match.group(1)
        return " ".join([digits[i:i+4] for i in range(0, 12, 4)])

    for i in range(len(tokens) - 2):
        digit_groups = [re.sub(r"\D", "", tokens[j]) for j in range(i, i + 3)]
        digits = "".join(digit_groups)
        if len(digits) == 12:
            return " ".join([digits[k:k+4] for k in range(0, 12, 4)])

    for i in range(len(tokens) - 2):
        combined = _normalize_aadhar_candidate(tokens[i] + tokens[i + 1] + tokens[i + 2])
        if len(combined) >= 12:
            candidate = re.sub(r"\D", "", combined)
            if len(candidate) == 12:
                return " ".join([candidate[k:k+4] for k in range(0, 12, 4)])
    return ""


def _find_gender(tokens):
    joined = " ".join(tokens).lower()
    if "female" in joined:
        return "Female"
    if "male" in joined:
        return "Male"

    for token in tokens:
        normalized = token.strip().lower()
        if "female" in normalized:
            return "Female"
        if "male" in normalized:
            return "Male"
        if normalized in {"m", "f"}:
            return "Male" if normalized == "m" else "Female"
        if "/" in normalized:
            if "female" in normalized:
                return "Female"
            if "male" in normalized:
                return "Male"

    for index, token in enumerate(tokens[:-1]):
        label = token.strip().lower()
        if label in {"gender", "sex"}:
            next_token = tokens[index + 1].strip().lower()
            if "female" in next_token:
                return "Female"
            if "male" in next_token:
                return "Male"
            if next_token in {"m", "f"}:
                return "Male" if next_token == "m" else "Female"
            if "/" in next_token:
                if "female" in next_token:
                    return "Female"
                if "male" in next_token:
                    return "Male"
    return ""


def _find_dob(tokens):
    for token in tokens:
        parsed = _parse_date(token)
        if parsed is not None:
            return parsed.strftime("%d/%m/%Y")
    for index, token in enumerate(tokens[:-1]):
        compact = re.sub(r"[^a-z]", "", token.lower())
        if compact in {"dob", "dateofbirth", "yearofbirth", "yob"}:
            parsed = _parse_date(tokens[index + 1])
            if parsed is not None:
                return parsed.strftime("%d/%m/%Y")
    return ""


def _find_name(tokens):
    for index, token in enumerate(tokens):
        normalized = token.strip()
        if ":" in normalized and "name" in normalized.lower():
            candidate = normalized.split(":", 1)[1].strip()
            if _looks_like_person_name(candidate):
                return candidate

    for index, token in enumerate(tokens):
        compact = re.sub(r"[^a-z]", "", token.lower())
        if compact == "name":
            name_candidate = []
            for next_token in tokens[index + 1:]:
                if not next_token.strip():
                    continue
                if _looks_like_person_name(next_token):
                    name_candidate.append(next_token)
                else:
                    break
                if len(name_candidate) >= 3:
                    break
            if name_candidate:
                return " ".join(name_candidate).strip()

    for index, token in enumerate(tokens):
        if re.search(r"\b(dob|date of birth|dateofbirth|yearofbirth|yob)\b", token, re.IGNORECASE):
            if index > 0 and _looks_like_person_name(tokens[index - 1]):
                return tokens[index - 1].strip()
            if index > 1 and _looks_like_person_name(tokens[index - 2]):
                return tokens[index - 2].strip()

    for index, token in enumerate(tokens):
        if re.search(r"\b(gender|sex|male|female)\b", token, re.IGNORECASE):
            if index > 0 and _looks_like_person_name(tokens[index - 1]):
                return tokens[index - 1].strip()
            if index > 1 and _looks_like_person_name(tokens[index - 2]):
                return tokens[index - 2].strip()

    best_name = ""
    for token in tokens:
        if _looks_like_person_name(token) and len(token.strip()) > len(best_name):
            best_name = token.strip()
    return best_name


def extract_information1(data_string):
    words = _split_text_tokens(data_string)
    extracted_info = {
        "ID": "",
        "Name": "",
        "Gender": "",
        "DOB": "",
        "ID Type": "AADHAR"
    }

    extracted_info["ID"] = _find_aadhar_id(words, data_string)
    extracted_info["Gender"] = _find_gender(words)
    extracted_info["DOB"] = _find_dob(words)
    extracted_info["Name"] = _find_name(words)

    return extracted_info

