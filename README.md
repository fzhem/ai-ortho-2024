# AI in Orthopaedics 2024 Hackathon

Welcome to the repository for the **[AI in Orthopaedics 2024 Hackathon](https://web.archive.org/web/20241217114842/https://www.boa.ac.uk/learning-and-events/ai-in-orthopaedics-and-msk-2024/hackathon-boa-ai-in-orthopaedics-and-msk-2024.html)**! This repository contains the code, datasets, and resources developed and submitted during the event.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Team Members](#team-members)
- [Acknowledgments](#acknowledgments)

## Overview

This project, developed during the AI in Orthopaedics 2024 Hackathon, uses AI to address orthopaedic challenges, focusing on detecting disc hernia via pelvic tilt, lumbar lordosis angle, sacral slope, and pelvic radius.

This project is a joint contribution from members in **[Southampton Emerging Therapies and Technologies (SETT) Centre](https://research.uhs.nhs.uk/about-us/facilities/southampton-emerging-therapies-and-technologies-sett-centre)**, the **[University of Southampton](https://www.southampton.ac.uk)**, and **Wessex Deanery**.

## Features

- Classifies disc hernia, spondylolisthesis, or normal cases.
- Custom code to handle scikit-learn Pipeline objects in downstream tasks
- Custom hierarchical model
- **Interactive Predictions**: A user-friendly front-end using Streamlit for predictions.
- **SHAP Explanations**: Provides explainable AI (XAI) outputs for model predictions using SHAP.

## Installation

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/fzhem/ai-ortho-2024.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ai-ortho-2024
   ```
3. Install the required dependencies using either of the following methods:
   - With `pip`:
     ```bash
     pip install -r requirements.txt
     ```
   - With `poetry`:
     ```bash
     poetry install
     ```

## Usage

To run the project, follow these steps:

1. The main code is in `exploratory.ipynb` file. This file uses custom code from `extended_pipeline.py` and `hierarchical_model.py`.
2. To run the front-end:
   ```bash
   streamlit run app.py
   ```

## Dataset

- **Source:** [UCI Vertebral Column](https://archive.ics.uci.edu/dataset/212/vertebral+column)
- The dataset includes 6 biomechanical attributes:
  - Pelvic Incidence
  - Pelvic Tilt
  - Lumbar Lordosis Angle
  - Sacral Slope
  - Pelvic Radius
  - Degree of Spondylolisthesis
- **Labels**: Normal, Hernia, Spondylolisthesis

## Team Members

In alphabetical order:
- **Ananya Pandey:** R&D Specialist Data Analyst
- **Faizan Hemotra:** R&D Specialist Data Analyst
- **Kehinde Makinde:** R&D Specialist Data Analyst
- **Lucy Bailey:** Wessex Deanery Trauma and Orthopaedic Specialist Registrar
- **Rory Ormiston:** Wessex Deanery Trauma and Orthopaedic Specialist Registrar, University of Southampton PhD Student

## Acknowledgments

We would like to thank:

- The organizers of the AI in Orthopaedics 2024 Hackathon for providing this opportunity.

---

For more details or questions, feel free to open an [issue](https://github.com/fzhem/ai-ortho-2024/issues).
