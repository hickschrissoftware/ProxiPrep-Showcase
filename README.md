ProxiPrep
Lead Developer: Chris Hicks

Target Platform: iOS/Android (Commercial Kitchen Integration)

Project Status: Integration Phase (Relational Database Migration)

Project Narrative
ProxiPrep was conceived as a solution to the "analog bottleneck" prevalent in commercial culinary environments. In high-volume kitchens, critical prep instructions are often relegated to handwritten notes and whiteboards, creating a data gap between physical inventory and digital Point-of-Sale (POS) systems. This project bridges that gap by implementing a custom computer vision pipeline designed to digitize shorthand culinary data and translate it into actionable inventory intelligence. By correlating historical prep volume with sales velocity, the system facilitates predictive forecasting aimed at reducing annual food waste by an estimated 20%.

Technical Implementation
Computer Vision & OCR Pre-processing
The primary engineering challenge involved the extraction of text from high-glare, non-uniform surfaces common in restaurant environments, such as stainless steel and damp cardboard. To address this, the scripts/ocr_engine.py module utilizes a multi-stage pre-processing workflow. I implemented Sauvola Binarization for adaptive thresholding, which ensures legibility across varied lighting conditions, and CLAHE (Contrast Limited Adaptive Histogram Equalization) to normalize contrast on low-visibility surfaces. This processed imagery is then fed into an ensemble voting architecture to maximize transcription accuracy for messy handwriting.

Relational Data Modeling & Shorthand Mapping
Following the database normalization principles established in DAD 220, ProxiPrep has transitioned from a flat JSON storage model to a robust SQLite relational schema. The core of the system is the shorthand_map table, which functions as a "Kitchen Thesaurus". This module uses fuzzy matching logic to translate over 650 unique shorthand terms (e.g., "Chx") into standardized inventory IDs. The "Self-Healing" infrastructure is designed to identify and flag unrecognized terms during the ingestion process, allowing the system to expand its library autonomously without manual script intervention.

Predictive Logic & Forecasting
The forecasting engine, located in scripts/predictor.py, serves as the system's analytical layer. It applies Applied Statistics to the digitized historical data to predict ingredient requirements for future shifts. This allows kitchen management to automate labor allocation and ingredient prep, ensuring that high-demand items are accounted for before morning rushes occur.

Academic Integration
This development effort serves as a practical application of the Computer Science curriculum at Southern New Hampshire University. Maintaining a 3.958 GPA, I have utilized frameworks from Discrete Mathematics (MAT 230) for algorithmic efficiency and Software Engineering for system design to ensure that ProxiPrep is not just a script, but a scalable industrial solution.

License: MIT
