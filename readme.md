# E-KYC Project

This is my E-KYC project — a personal implementation of an identity verification system that combines OCR, face recognition, and database checks to automate KYC processing for Aadhar and PAN cards.

## Overview

I built this project to let users upload an ID card and a face image, then verify the ID details and match the face from the ID with the uploaded photo.

Currently supported card types:
- Aadhar
- PAN

## What it does

- **Face verification**: Extracts the face from the uploaded ID card using Haarcascade and compares it against the uploaded photograph.
- **OCR extraction**: Reads text from the ID card using EasyOCR and converts it into structured fields.
- **Duplicate detection**: Checks if the user already exists in the database before saving a new record.
- **Face embeddings**: Uses DeepFace FaceNet embeddings to support face matching.

## How it works

1. Upload an Aadhar or PAN card image.
2. Upload a face photo.
3. The app verifies the card face against the uploaded face.
4. If verification succeeds, it extracts ID data using OCR.
5. The extracted data is validated and stored securely if no duplicate is found.

## Technologies used

- Python
- Streamlit
- EasyOCR
- DeepFace
- Haarcascade
- MySQL

## Improvements I’m working on

- Live face detection through device camera instead of requiring a separate photo.
- Better data privacy by hashing sensitive fields before storing them in the database.

## Prerequisites

- Python 3.12 (or compatible version)
- MySQL server






This README is written from my perspective to explain the project clearly for GitHub viewers and contributors.



