import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import json
import base64
from io import BytesIO

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not found. Image quality assessment will be limited.")

st.title("HER2 Breast Cancer Risk Detector (Enhanced)")

# Initialize session state for historical tracking
if 'test_history' not in st.session_state:
    st.session_state.test_history = []
if 'family_history' not in st.session_state:
    st.session_state.family_history = {}
if 'medications' not in st.session_state:
    st.session_state.medications = []
if 'appointments' not in st.session_state:
    st.session_state.appointments = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0

# Create tabs for analysis, information, symptoms, and new features
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "Single Image Analysis", 
    "Multi-Image Comparison", 
    "Historical Tracking", 
    "Family History", 
    "Support & Resources",
    "About HER2 & Breast Cancer", 
    "Symptoms Assessment",
    "AI Health Assistant",
    "Medication Tracker",
    "Appointment Scheduler",
    "Educational Quiz"
])

def assess_image_quality(image):
    """Assess the quality of the uploaded image"""
    image_array = np.array(image)

    # Check image size
    height, width = image_array.shape[:2]
    size_score = min(100, (height * width) / 1000)  # Normalize to 100

    if OPENCV_AVAILABLE:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Check for blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100, laplacian_var / 10)  # Normalize to 100

        # Check brightness
        brightness = np.mean(gray)
        if 50 <= brightness <= 200:
            brightness_score = 100
        else:
            brightness_score = max(0, 100 - abs(brightness - 125) / 2)

        # Check contrast
        contrast = np.std(gray)
        contrast_score = min(100, contrast * 2)

        # Overall score
        overall_score = (size_score * 0.2 + blur_score * 0.4 + brightness_score * 0.2 + contrast_score * 0.2)
    else:
        # Fallback: simplified analysis without OpenCV
        # Convert to grayscale manually
        gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])

        # Check brightness
        brightness = np.mean(gray)
        if 50 <= brightness <= 200:
            brightness_score = 100
        else:
            brightness_score = max(0, 100 - abs(brightness - 125) / 2)

        # Check contrast
        contrast = np.std(gray)
        contrast_score = min(100, contrast * 2)

        # Simplified blur detection (edge variance)
        edges = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        blur_score = min(100, edges * 5)

        # Overall score
        overall_score = (size_score * 0.3 + blur_score * 0.3 + brightness_score * 0.2 + contrast_score * 0.2)

    if overall_score >= 80:
        return {"quality": "good", "reason": "Image quality is excellent for analysis", "score": int(overall_score)}
    elif overall_score >= 50:
        return {"quality": "fair", "reason": "Image quality is acceptable but could be improved", "score": int(overall_score)}
    else:
        return {"quality": "poor", "reason": "Image quality needs improvement", "score": int(overall_score)}

def detect_strip_orientation(image):
    """Detect if the strip is properly oriented"""
    image_array = np.array(image)
    height, width = image_array.shape[:2]

    # Analyze aspect ratio
    aspect_ratio = width / height

    if aspect_ratio > 2:
        orientation = "horizontal"
        confidence = "high"
    elif aspect_ratio < 0.5:
        orientation = "vertical"
        confidence = "high"
    elif 1.5 < aspect_ratio <= 2:
        orientation = "horizontal"
        confidence = "medium"
    elif 0.5 <= aspect_ratio < 0.67:
        orientation = "vertical"
        confidence = "medium"
    else:
        orientation = "square/unclear"
        confidence = "low"

    # Check for strip-like features
    if OPENCV_AVAILABLE:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        # Look for horizontal lines (typical for strips)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 200) / horizontal_lines.size

        # Look for vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, vertical_kernel)
        vertical_score = np.sum(vertical_lines > 200) / vertical_lines.size

        if horizontal_score > vertical_score * 1.5:
            detected_orientation = "horizontal"
        elif vertical_score > horizontal_score * 1.5:
            detected_orientation = "vertical"
        else:
            detected_orientation = "unclear"
    else:
        detected_orientation = "unclear"

    return {
        "orientation": orientation,
        "detected_orientation": detected_orientation,
        "confidence": confidence,
        "aspect_ratio": round(aspect_ratio, 2),
        "recommendation": get_orientation_recommendation(orientation, detected_orientation)
    }

def get_orientation_recommendation(orientation, detected):
    if orientation == "square/unclear" or detected == "unclear":
        return "⚠️ Strip orientation unclear. Please ensure the test strip is clearly visible and properly framed."
    elif orientation == "vertical" and detected == "horizontal":
        return "🔄 Consider rotating the image 90° - strip appears to be oriented vertically but may work better horizontally."
    elif orientation == "horizontal" and detected == "vertical":
        return "🔄 Consider rotating the image 90° - strip appears to be oriented horizontally but may work better vertically."
    else:
        return "✅ Strip orientation looks good for analysis."

def classify_risk(percentage):
    if percentage >= 70:
        return "⚠️ High Risk (Very dark red)", "high"
    elif percentage >= 40:
        return "⚠️ Moderate Risk (Medium red)", "moderate"
    elif percentage >= 15:
        return "🟡 Low Risk (Faint red/pink)", "low"
    else:
        return "✅ No Risk (Light pink to no color)", "no_risk"

def get_medical_guidance(risk_level, percentage, financial_assistance=False):
    base_guidance = ""
    financial_section = ""

    if risk_level == "high":
        base_guidance = f"""
### 🚨 **URGENT MEDICAL ATTENTION REQUIRED**

**Your HER2 protein level is {percentage}% - This indicates HIGH RISK**

#### Immediate Actions:
- **Contact your doctor TODAY** or visit the emergency room
- **Schedule immediate screening** including:
  - Mammogram
  - Ultrasound
  - Possible biopsy
- **Bring this test result** to your healthcare provider
"""
        if financial_assistance:
            financial_section = """
#### 💰 **Financial Assistance Options (Philippines):**
- **Barangay Health Station:** Free initial consultation and referral
- **Government Hospital Charity Service:** For indigent patients (bring Certificate of Indigency)
- **PhilHealth Z-Benefit Package:** Covers cancer treatment costs
- **PCSO Medical Assistance:** Financial aid for cancer treatments
- **I Can Serve Foundation:** Free mammography and breast cancer screening
- **Malasakit Centers:** One-stop medical assistance at government hospitals
- **DSWD Medical Assistance:** For qualified indigent families
"""
    elif risk_level == "moderate":
        base_guidance = f"""
### ⚠️ **MEDICAL CONSULTATION RECOMMENDED**

**Your HER2 protein level is {percentage}% - This indicates MODERATE RISK**

#### Recommended Actions:
- **Schedule appointment with your doctor within 1-2 weeks**
- **Request breast screening** including:
  - Clinical breast examination
  - Mammogram (if over 40 or as recommended)
  - Possible ultrasound
"""
        if financial_assistance:
            financial_section = """
#### 💰 **Affordable Care Options (Philippines):**
- **Provincial/City Health Offices:** Low-cost screening programs
- **I Can Serve Foundation:** Free breast cancer screening and mammography
- **Rural Health Units:** Basic health services in municipalities
- **PhilHealth Outpatient Benefit Package:** Covers screening costs
- **Mobile Health Units (DOH):** Free community screenings
- **Local Government Medical Assistance:** Contact your Mayor's office
"""
    elif risk_level == "low":
        base_guidance = f"""
### 🟡 **ROUTINE MONITORING SUGGESTED**

**Your HER2 protein level is {percentage}% - This indicates LOW RISK**

#### Recommended Actions:
- **Schedule routine check-up** with your healthcare provider
- **Continue regular screening** as per age guidelines
"""
        if financial_assistance:
            financial_section = """
#### 💰 **Low-Cost Prevention Options (Philippines):**
- **Barangay Health Fairs:** Free basic health screenings
- **DOH Health Programs:** National preventive care initiatives
- **School Health Clinics:** University health centers
- **Corporate Health Programs:** Company-sponsored health checkups
- **Philippine Cancer Society Screening:** Free detection programs
"""
    else:
        base_guidance = f"""
### ✅ **NO IMMEDIATE CONCERN**

**Your HER2 protein level is {percentage}% - This indicates NO SIGNIFICANT RISK**

#### Current Status:
- **No immediate medical action required**
- **Continue routine preventive care**
"""
        if financial_assistance:
            financial_section = """
#### 💰 **Preventive Care Resources (Philippines):**
- **Annual Barangay Health Checkups:** Free routine health assessments
- **PhilHealth Preventive Care Package:** Regular checkups covered
- **Rural Health Units:** Routine care for all residents
- **Company Health Programs:** Employee health benefits and checkups
"""

    return base_guidance + financial_section

def analyze_image(image, image_name="Image"):
    # Image quality assessment
    quality_assessment = assess_image_quality(image)

    # Strip orientation detection
    orientation_info = detect_strip_orientation(image)

    # Color analysis
    image_array = np.array(image)
    red_channel = image_array[:, :, 0]
    avg_red = int(np.mean(red_channel))

    # Convert to percentage scale (0-100%)
    percentage = min(100, max(0, int((avg_red / 255) * 100)))

    risk_text, risk_level = classify_risk(percentage)

    return {
        "name": image_name,
        "percentage": percentage,
        "avg_red": avg_red,
        "risk_text": risk_text,
        "risk_level": risk_level,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "quality": quality_assessment,
        "orientation": orientation_info
    }

def create_shareable_report(result, family_history=None, financial_assistance=False):
    """Create a shareable report for healthcare providers"""
    report = f"""
# HER2 Breast Cancer Analysis Report

**Patient Information:**
- Analysis Date: {result['timestamp']}
- Test Result: {result['percentage']}% HER2 protein level
- Risk Assessment: {result['risk_text']}

**Technical Analysis:**
- Image Quality: {result['quality']['quality'].upper()} (Score: {result['quality']['score']}/100)
- Quality Notes: {result['quality']['reason']}
- Strip Orientation: {result['orientation']['orientation'].upper()}
- Orientation Confidence: {result['orientation']['confidence'].upper()}

**Family History:**
"""
    if family_history:
        for relation, details in family_history.items():
            if details['has_history']:
                report += f"- {relation}: {details['cancer_type']} at age {details['age']}\n"
    else:
        report += "- No family history data provided\n"

    report += f"""
**Clinical Recommendation:**
{get_medical_guidance(result['risk_level'], result['percentage'], financial_assistance)}

**Important Notes:**
- This is a preliminary screening result based on image analysis
- Professional medical evaluation is required for definitive diagnosis
- Please bring this report to your healthcare provider
- Results should be confirmed with laboratory testing

**Contact Information:**
- Patient Name: [Please fill in]
- Phone: [Please fill in]
- Emergency Contact: [Please fill in]

---
Report generated by HER2 Breast Cancer Risk Detector
For educational and screening purposes only - Not a medical diagnosis
"""
    return report

# Single Image Analysis Tab
with tab1:
    st.markdown("### Upload and Analyze Your LFA Strip Image")

    # Financial assistance checkbox
    financial_help = st.checkbox("💰 I need information about low-cost/free healthcare options")

    uploaded_file = st.file_uploader("Upload your LFA strip image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Display image with analysis
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Strip", use_container_width=True)

        with col2:
            result = analyze_image(image, uploaded_file.name)

            # Image Quality Assessment
            st.markdown("#### 📊 Image Quality Assessment")
            quality = result['quality']
            if quality['quality'] == 'good':
                st.success(f"✅ {quality['reason']} (Score: {quality['score']}/100)")
            elif quality['quality'] == 'fair':
                st.warning(f"⚠️ {quality['reason']} (Score: {quality['score']}/100)")
            else:
                st.error(f"❌ {quality['reason']} (Score: {quality['score']}/100)")
                st.info("**Tips for better image quality:**\n- Ensure good lighting\n- Hold camera steady\n- Get closer to the strip\n- Avoid shadows and glare")

            # Strip Orientation
            st.markdown("#### 🔄 Strip Orientation")
            orientation = result['orientation']
            st.info(f"**Detected:** {orientation['orientation']} (Confidence: {orientation['confidence']})")
            st.write(orientation['recommendation'])

        # Risk Analysis Results
        st.markdown("---")
        st.write(f"📊 **HER2 Protein Level:** {result['percentage']}%")
        st.markdown(f"### 🔍 Risk Assessment: {result['risk_text']}")

        # Display medical guidance with financial assistance if needed
        medical_guidance = get_medical_guidance(result['risk_level'], result['percentage'], financial_help)
        st.markdown(medical_guidance)

        # Add to history
        if st.button("💾 Save to History"):
            st.session_state.test_history.append(result)
            st.success("Result saved to your test history!")

        # Doctor sharing
        st.markdown("### 👩‍⚕️ Share with Healthcare Provider")
        if st.button("📄 Generate Doctor Report"):
            report = create_shareable_report(result, st.session_state.family_history, financial_help)

            # Create download button
            st.download_button(
                label="📥 Download Report for Doctor",
                data=report,
                file_name=f"her2_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

            # Display shareable link info
            st.info("""
            **Sharing Options:**
            - Download this report and bring it to your appointment
            - Email the report to your healthcare provider
            - Print the report for your medical records
            - Show the report on your phone during consultation
            """)

        st.markdown("---")
        st.markdown("ℹ️ **Note:** This is a simulated result for educational purposes only. Not a medical diagnosis. Always consult healthcare professionals.")

# Multi-Image Analysis Tab (keeping existing functionality)
with tab2:
    st.markdown("### Upload Multiple Strip Images for Comparison")

    uploaded_files = st.file_uploader(
        "Upload multiple LFA strip images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown(f"**Analyzing {len(uploaded_files)} images...**")

        results = []
        cols = st.columns(min(3, len(uploaded_files)))

        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file).convert("RGB")
            result = analyze_image(image, uploaded_file.name)
            results.append(result)

            # Display images in columns
            with cols[i % 3]:
                st.image(image, caption=f"{uploaded_file.name}", use_container_width=True)
                st.write(f"**HER2 Level:** {result['percentage']}%")
                st.write(f"**Risk:** {result['risk_text']}")
                st.write(f"**Quality:** {result['quality']['quality']}")

        # Summary table
        st.markdown("### 📊 Comparison Summary")
        df = pd.DataFrame(results)

        # Create a formatted display table
        display_df = df[['name', 'percentage', 'risk_text', 'timestamp']].copy()
        display_df.columns = ['Image Name', 'HER2 Level (%)', 'Risk Assessment', 'Analysis Time']

        st.dataframe(display_df, use_container_width=True)

        # Save all to history
        if st.button("💾 Save All to History"):
            st.session_state.test_history.extend(results)
            st.success(f"All {len(results)} results saved to your test history!")

# Historical Tracking Tab
with tab3:
    st.markdown("# 📈 Historical Test Results")

    if st.session_state.test_history:
        # Convert to DataFrame for analysis
        history_df = pd.DataFrame(st.session_state.test_history)

        # Display trend chart
        st.markdown("### HER2 Level Trends Over Time")
        chart_data = history_df[['timestamp', 'percentage']].copy()
        chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
        chart_data = chart_data.set_index('timestamp')
        st.line_chart(chart_data)

        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tests", len(history_df))
        with col2:
            st.metric("Average Level", f"{history_df['percentage'].mean():.1f}%")
        with col3:
            st.metric("Latest Result", f"{history_df.iloc[-1]['percentage']}%")
        with col4:
            high_risk_count = len(history_df[history_df['risk_level'] == 'high'])
            st.metric("High Risk Tests", high_risk_count)

        # Detailed history table
        st.markdown("### Detailed Test History")
        display_history = history_df[['timestamp', 'name', 'percentage', 'risk_text']].copy()
        display_history.columns = ['Date/Time', 'Image', 'HER2 Level (%)', 'Risk Assessment']
        st.dataframe(display_history.sort_values('timestamp', ascending=False), use_container_width=True)

        # Export options
        if st.button("📄 Export Complete History"):
            history_report = "# Complete HER2 Test History\n\n"
            for _, test in history_df.iterrows():
                history_report += f"**{test['timestamp']}**\n"
                history_report += f"- Image: {test['name']}\n"
                history_report += f"- HER2 Level: {test['percentage']}\n"
                history_report += f"- Risk: {test['risk_text']}\n"
                history_report += f"- Quality: {test['quality']['quality']}\n\n"

            st.download_button(
                label="📥 Download History Report",
                data=history_report,
                file_name=f"her2_test_history_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

        # Clear history option
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.test_history = []
            st.success("Test history cleared!")
            st.rerun()
    else:
        st.info("No test history available yet. Complete some analyses to see your trends here!")

# Family History Tab
with tab4:
    st.markdown("# 👨‍👩‍👧‍👦 Family Cancer History")
    st.markdown("Tracking family cancer history helps assess your overall risk profile.")

    family_members = ["Mother", "Father", "Sister", "Brother", "Maternal Grandmother", 
                     "Maternal Grandfather", "Paternal Grandmother", "Paternal Grandfather",
                     "Aunt (Mother's side)", "Aunt (Father's side)", "Uncle (Mother's side)", "Uncle (Father's side)"]

    for member in family_members:
        st.markdown(f"### {member}")
        col1, col2, col3 = st.columns([2, 2, 2])

        with col1:
            has_history = st.checkbox(f"Had cancer", key=f"{member}_history")

        if has_history:
            with col2:
                cancer_type = st.selectbox(
                    "Cancer type",
                    ["Breast Cancer", "Ovarian Cancer", "Prostate Cancer", "Colorectal Cancer", 
                     "Lung Cancer", "Other Cancer"],
                    key=f"{member}_type"
                )
            with col3:
                age = st.number_input(f"Age at diagnosis", min_value=0, max_value=100, key=f"{member}_age")

            st.session_state.family_history[member] = {
                "has_history": True,
                "cancer_type": cancer_type,
                "age": age
            }
        else:
            st.session_state.family_history[member] = {"has_history": False}

    # Family risk assessment
    if st.button("📊 Assess Family Risk"):
        family_cancer_count = sum(1 for member in st.session_state.family_history.values() if member.get('has_history', False))
        breast_cancer_count = sum(1 for member in st.session_state.family_history.values() 
                                if member.get('has_history', False) and member.get('cancer_type') == 'Breast Cancer')

        st.markdown("### Family Risk Assessment")
        if breast_cancer_count >= 2:
            st.error("⚠️ **HIGH FAMILY RISK** - Multiple breast cancer cases in family. Consider genetic counseling.")
        elif breast_cancer_count == 1:
            st.warning("🟡 **MODERATE FAMILY RISK** - One breast cancer case in family. Discuss with doctor about early screening.")
        elif family_cancer_count >= 3:
            st.warning("🟡 **MODERATE FAMILY RISK** - Multiple cancer cases in family. Consider genetic counseling.")
        else:
            st.success("✅ **LOW FAMILY RISK** - No significant family cancer history detected.")

# Support & Resources Tab
with tab5:
    st.markdown("# 🤝 Support Groups & Resources")

    # Support Group Finder
    st.markdown("## 🔍 Find Support Groups")

    col1, col2 = st.columns(2)
    with col1:
        location = st.text_input("Enter your city or ZIP code", placeholder="e.g., New York, NY or 10001")
    with col2:
        support_type = st.selectbox("Type of support needed", [
            "General Breast Cancer Support",
            "HER2-Positive Support",
            "Young Adults with Cancer",
            "Cancer Caregivers",
            "Financial Assistance",
            "Online Support Groups"
        ])

    if st.button("🔍 Find Support Groups"):
        st.markdown("### 📍 Philippines Support Resources")

        # Philippines-specific support group results
        st.markdown("""
        #### **National Organizations (Philippines):**

        **🌟 Philippine Cancer Society (PCS)**
        - Website: philcancer.org.ph
        - Hotline: (02) 8-929-2376
        - Main Office: Rm 301 EGI Taft Tower, UN Ave, Manila
        - Free cancer detection programs
        - Support groups and counseling services

        **🌟 I Can Serve Foundation**
        - Website: icanserve.ph
        - Hotline: (02) 8-524-1221
        - Breast cancer awareness and support
        - Free mammography programs
        - Patient navigation services

        **🌟 Department of Health (DOH) Cancer Control Program**
        - Website: doh.gov.ph
        - Philippine Cancer Control Program
        - Regional Medical Centers with oncology services
        - Public health insurance coverage (PhilHealth)

        **🌟 Philippine Society of Medical Oncology (PSMO)**
        - Website: psmo.org.ph
        - Directory of oncologists nationwide
        - Treatment guidelines and protocols
        - Patient education resources

        **🌟 Kythe Foundation**
        - Website: kythe.org
        - Hotline: (02) 8-997-KYTHE (59843)
        - Support for cancer patients and families
        - Creative arts therapy programs

        #### **Regional Cancer Centers:**

        **🏥 Philippine General Hospital (PGH)**
        - Cancer Institute, Manila
        - Charity service available
        - Contact: (02) 8-554-8400

        **🏥 National Kidney and Transplant Institute (NKTI)**
        - Cancer Center, Quezon City
        - Contact: (02) 8-981-0300

        **🏥 Davao Medical School Foundation Hospital**
        - Cancer Center, Davao
        - Contact: (082) 226-4589

        **🏥 Cebu Velez General Hospital**
        - Cancer Center, Cebu
        - Contact: (032) 254-5555

        #### **Online Support Communities (Filipino):**

        **💬 Facebook Groups:**
        - "Breast Cancer Warriors Philippines"
        - "Cancer Fighters Philippines Support Group"
        - "Mga Bayaning Nakikipaglaban sa Cancer"
        - "Philippine Cancer Survivors Network"

        **💬 Viber Communities:**
        - "PH Cancer Support Network"
        - "Breast Cancer Warriors PH"

        **💬 Telegram Groups:**
        - "Cancer Support Philippines"
        - "Mga Mandirigma Laban sa Cancer"
        """)

    # Financial Assistance Resources
    st.markdown("## 💰 Financial Assistance Resources")

    financial_resources = {
        "Free/Low-Cost Healthcare (Philippines)": [
            "**Rural Health Units (RHUs)** - Free basic healthcare in barangays",
            "**Barangay Health Stations** - Primary healthcare at community level",
            "**Government Hospitals** - Charity service for indigent patients",
            "**DOH Regional Medical Centers** - Free cancer screening programs",
            "**Philippine General Hospital (PGH)** - Charity service for qualified patients"
        ],
        "Government Assistance Programs": [
            "**PhilHealth** - National health insurance program",
            "**PCSO Medical Assistance Program** - Financial aid for treatments",
            "**Department of Social Welfare (DSWD)** - Medical assistance for indigents",
            "**Malasakit Centers** - One-stop shop for medical assistance",
            "**Mayor's Office/LGU Medical Assistance** - Local government aid programs"
        ],
        "Foundation and NGO Support": [
            "**I Can Serve Foundation** - Free mammography and breast cancer programs",
            "**Philippine Cancer Society** - Patient assistance and support",
            "**Kythe Foundation** - Support for cancer patients and families",
            "**Rotary Club Cancer Assistance** - Local chapter support programs",
            "**Lions Club Medical Mission** - Community health programs"
        ],
        "Insurance and Coverage": [
            "**PhilHealth Universal Healthcare** - Mandatory coverage for all Filipinos",
            "**Indigent Program (PhilHealth)** - Free coverage for poor families",
            "**Senior Citizen Benefits** - 20% discount on medicines and services",
            "**PWD Benefits** - Discounts for persons with disabilities",
            "**Health Maintenance Organizations (HMOs)** - Private health insurance"
        ]
    }

    for category, resources in financial_resources.items():
        with st.expander(f"📋 {category}"):
            for resource in resources:
                st.markdown(f"• {resource}")

    # Emergency Action Plan
    st.markdown("## 🚨 What to Do If You Can't Afford Immediate Care")

    st.error("""
    **If you have concerning symptoms but cannot afford immediate care:**

    1. **Visit your Barangay Health Station** - Free initial consultation and referral
    2. **Go to the nearest Government Hospital** - Cannot refuse emergency cases
    3. **Contact PCSO Medical Assistance** - Call (02) 8-304-5110 for immediate help
    4. **Apply for PhilHealth** - Coverage can be processed quickly for emergencies
    5. **Visit your Local Health Department** - Provincial/city health offices provide free services
    6. **Contact Malasakit Center** - One-stop assistance at major government hospitals
    7. **Reach out to I Can Serve Foundation** - (02) 8-524-1221 for breast cancer assistance

    **Remember: Early treatment is often less expensive than waiting until cancer advances.**
    **Sa Pilipinas, may mga libreng serbisyo para sa mga nangangailangan.**
    """)

    # Quick Contact Information
    st.markdown("## 📞 Quick Contact Information")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Emergency Hotlines:**
        - Emergency: 911
        - Philippine Cancer Society: (02) 8-929-2376
        - I Can Serve Foundation: (02) 8-524-1221
        - DOH Hotline: (02) 8-651-7800 local 1149-1150
        """)

    with col2:
        st.info("""
        **Financial Assistance:**
        - PCSO Medical Assistance: (02) 8-304-5110
        - DSWD Hotline: (02) 8-931-8101
        - PhilHealth: (02) 8-441-7442
        - Malasakit Center: Available at government hospitals
        """)

# About HER2 & Breast Cancer Tab (keeping existing content)
with tab6:
    st.markdown("# About HER2 Protein and Breast Cancer")

    st.markdown("### What is HER2?")
    st.markdown("""
    **HER2 (Human Epidermal Growth Factor Receptor 2)** is a protein that can be found on the surface of some breast cancer cells. 
    When HER2 is overexpressed, it can promote the growth of cancer cells.

    - **HER2-positive breast cancer:** About 15-20% of breast cancers are HER2-positive
    - **HER2-negative breast cancer:** The majority of breast cancers are HER2-negative
    - **Testing importance:** HER2 status helps determine the most effective treatment options
    """)

    st.markdown("### How This Test Works")
    st.markdown("""
    This simulated test mimics a **Lateral Flow Assay (LFA)** that detects HER2 protein levels on a 0-100% scale:

    🔴 **70-100%:** High HER2 protein levels (Dark Red) - **HIGH RISK** - Urgent medical attention required
    🟠 **40-69%:** Moderate HER2 protein levels (Medium Red) - **MODERATE RISK** - Medical consultation recommended  
    🟡 **15-39%:** Low HER2 protein levels (Faint Red/Pink) - **LOW RISK** - Routine monitoring suggested
    ⚪ **0-14%:** No significant HER2 detected (Light Pink/No Color) - **NO RISK** - Continue regular screening
    """)

    st.markdown("### Treatment Options for HER2-Positive Cancer")
    st.markdown("""
    - **Targeted Therapy:** Drugs like Herceptin (trastuzumab) specifically target HER2
    - **Chemotherapy:** Often combined with targeted therapy
    - **Surgery:** Lumpectomy or mastectomy depending on stage
    - **Radiation Therapy:** May be recommended after surgery
    - **Hormone Therapy:** If the cancer is also hormone receptor-positive
    """)

# Symptoms Assessment Tab (keeping existing content)
with tab7:
    st.markdown("# Breast Cancer Symptoms Assessment")

    st.markdown("### Common Breast Cancer Symptoms")
    st.markdown("""
    **Physical Changes:**
    - A new lump or mass in the breast or underarm area
    - Swelling of all or part of a breast
    - Skin dimpling or puckering
    - Breast or nipple pain
    - Nipple retraction (turning inward)
    - Nipple discharge (other than breast milk)
    - Red, dry, flaking, or thickened nipple or breast skin
    - Change in breast size or shape

    **Important Notes:**
    - Most breast lumps are NOT cancer
    - Changes can occur due to hormones, aging, or benign conditions
    - Early detection significantly improves treatment outcomes
    """)

    st.markdown("### Self-Assessment Checklist")

    symptoms = [
        "New lump or thickening in breast or underarm",
        "Change in breast size or shape",
        "Dimpling or puckering of breast skin",
        "Nipple discharge (not breast milk)",
        "Nipple turning inward",
        "Red, scaly, or thickened nipple/breast skin",
        "Breast or nipple pain that doesn't go away"
    ]

    selected_symptoms = []
    for symptom in symptoms:
        if st.checkbox(symptom):
            selected_symptoms.append(symptom)

    if selected_symptoms:
        st.markdown("### ⚠️ You have indicated the following symptoms:")
        for symptom in selected_symptoms:
            st.write(f"• {symptom}")

        st.markdown("### 🏥 **Immediate Action Required**")
        st.error("""
        **Contact your healthcare provider immediately if you experience any of these symptoms.**

        Early detection saves lives. Don't wait - schedule an appointment today.
        """)

    st.markdown("### When to Seek Medical Care")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🚨 Seek Immediate Care If:")
        st.markdown("""
        - Any new breast lump or mass
        - Nipple discharge with blood
        - Sudden breast swelling
        - Skin changes (dimpling, puckering)
        - Persistent breast pain
        - Any concerning breast changes
        """)

    with col2:
        st.markdown("#### 📅 Schedule Regular Checkups:")
        st.markdown("""
        - **Ages 20-39:** Clinical exam every 1-3 years
        - **Ages 40+:** Annual mammogram + clinical exam
        - **High risk:** Follow doctor's recommendations
        - **Family history:** Earlier/more frequent screening
        """)

    st.markdown("### Where to Get Help")
    st.markdown("""
    #### Healthcare Providers to Contact:
    - **Primary Care Physician:** Your first point of contact
    - **Gynecologist:** Specializes in women's health
    - **Oncologist:** Cancer specialist (if cancer is suspected/diagnosed)
    - **Breast Specialist:** Focuses specifically on breast health

    #### Emergency Services:
    - **Emergency Room:** For severe symptoms or urgent concerns
    - **Urgent Care:** For non-emergency but concerning symptoms
    - **Telehealth:** Many providers offer virtual consultations

    #### Support Resources:
    - **National Cancer Institute:** 1-800-4-CANCER (1-800-422-6237)
    - **American Cancer Society:** cancer.org
    - **Breast Cancer Support Groups:** Local and online communities
    - **Financial Assistance Programs:** For treatment costs
    """)

    st.info("""
    **Remember:** This tool is for educational purposes only and cannot replace professional medical advice. 
    Always consult with qualified healthcare providers for proper diagnosis and treatment.
    """)

# AI Health Assistant Tab
with tab8:
    st.markdown("# 🤖 AI Health Assistant")
    st.markdown("Ask questions about breast cancer, HER2, symptoms, or treatment options.")

    # Chat interface
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}..."):
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")

    # New question input
    st.markdown("### Ask a Question")
    user_question = st.text_area("What would you like to know about breast cancer or HER2?", 
                                placeholder="e.g., What are the side effects of HER2-targeted therapy?")

    if st.button("🔍 Get Answer") and user_question:
        # Simulated AI responses based on common questions
        responses = {
            "side effects": "Common side effects of HER2-targeted therapy include fatigue, nausea, diarrhea, and potential heart problems. Regular monitoring is important during treatment.",
            "treatment": "HER2-positive breast cancer treatment typically includes targeted therapy (like Herceptin), chemotherapy, surgery, and possibly radiation therapy.",
            "diet": "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins can support your health during treatment. Limit processed foods and alcohol.",
            "exercise": "Light to moderate exercise during treatment can help reduce fatigue and improve mood. Always consult your doctor before starting any exercise program.",
            "prognosis": "HER2-positive breast cancer prognosis has improved significantly with targeted therapies. Early detection and appropriate treatment lead to better outcomes.",
            "pregnancy": "HER2-targeted therapies can affect pregnancy. Discuss family planning with your oncologist before starting treatment.",
            "recurrence": "Regular follow-up care and monitoring help detect any recurrence early. Follow your doctor's surveillance schedule closely."
        }

        # Simple keyword matching for response
        answer = "I understand you're asking about breast cancer. While I can provide general information, it's important to discuss your specific situation with your healthcare provider. Here's some general information: "
        
        question_lower = user_question.lower()
        for keyword, response in responses.items():
            if keyword in question_lower:
                answer += response
                break
        else:
            answer += "For specific medical questions, please consult with your oncologist or healthcare team who can provide personalized advice based on your medical history."

        # Add to chat history
        st.session_state.chat_history.append((user_question, answer))
        
        st.success("Answer generated!")
        st.markdown(f"**Answer:** {answer}")

    # Common questions
    st.markdown("### Common Questions")
    
    # Define responses for common questions
    common_responses = {
        "What are the side effects of HER2 treatment?": "Common side effects of HER2-targeted therapy include fatigue, nausea, diarrhea, and potential heart problems. Regular monitoring is important during treatment.",
        "How often should I get tested?": "Testing frequency depends on your treatment stage. During active treatment, blood tests may be weekly or bi-weekly. After treatment, follow-up testing is typically every 3-6 months initially, then annually.",
        "Can I exercise during treatment?": "Light to moderate exercise during treatment can help reduce fatigue and improve mood. Always consult your doctor before starting any exercise program during cancer treatment.",
        "What foods should I eat?": "A balanced diet rich in fruits, vegetables, whole grains, and lean proteins can support your health during treatment. Limit processed foods and alcohol. Stay hydrated and consider small, frequent meals if experiencing nausea.",
        "What is the prognosis for HER2-positive cancer?": "HER2-positive breast cancer prognosis has improved significantly with targeted therapies like Herceptin. Early detection and appropriate treatment lead to better outcomes. Five-year survival rates are high when caught early."
    }
    
    common_questions = [
        "What are the side effects of HER2 treatment?",
        "How often should I get tested?",
        "Can I exercise during treatment?",
        "What foods should I eat?",
        "What is the prognosis for HER2-positive cancer?"
    ]

    for question in common_questions:
        if st.button(f"❓ {question}", key=f"common_{question}"):
            answer = common_responses.get(question, "Please consult your healthcare provider for specific medical advice.")
            st.session_state.chat_history.append((question, answer))
            st.rerun()

# Medication Tracker Tab
with tab9:
    st.markdown("# 💊 Medication Tracker")
    st.markdown("Keep track of your medications, dosages, and side effects.")

    # Add new medication
    st.markdown("### Add New Medication")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        med_name = st.text_input("Medication Name", placeholder="e.g., Herceptin")
    with col2:
        med_dosage = st.text_input("Dosage", placeholder="e.g., 600mg")
    with col3:
        med_frequency = st.selectbox("Frequency", ["Once daily", "Twice daily", "Three times daily", "Weekly", "Bi-weekly", "Monthly", "As needed"])

    med_start_date = st.date_input("Start Date")
    med_notes = st.text_area("Notes/Side Effects", placeholder="Any side effects or special instructions...")

    if st.button("➕ Add Medication"):
        new_med = {
            "name": med_name,
            "dosage": med_dosage,
            "frequency": med_frequency,
            "start_date": med_start_date.strftime("%Y-%m-%d"),
            "notes": med_notes,
            "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.medications.append(new_med)
        st.success(f"Added {med_name} to your medication list!")

    # Display current medications
    if st.session_state.medications:
        st.markdown("### Current Medications")
        
        for i, med in enumerate(st.session_state.medications):
            with st.expander(f"💊 {med['name']} - {med['dosage']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Frequency:** {med['frequency']}")
                    st.write(f"**Start Date:** {med['start_date']}")
                with col2:
                    st.write(f"**Notes:** {med['notes']}")
                    if st.button(f"🗑️ Remove", key=f"remove_med_{i}"):
                        st.session_state.medications.pop(i)
                        st.rerun()

        # Medication reminders
        st.markdown("### 🔔 Today's Medication Schedule")
        current_time = datetime.now().strftime("%H:%M")
        st.info(f"Current time: {current_time}")
        
        for med in st.session_state.medications:
            if "daily" in med['frequency'].lower():
                st.write(f"⏰ {med['name']} ({med['dosage']}) - {med['frequency']}")

    else:
        st.info("No medications added yet. Add your medications above to start tracking.")

# Appointment Scheduler Tab
with tab10:
    st.markdown("# 📅 Appointment Scheduler")
    st.markdown("Track your medical appointments and reminders.")

    # Add new appointment
    st.markdown("### Schedule New Appointment")
    col1, col2 = st.columns(2)
    
    with col1:
        appt_type = st.selectbox("Appointment Type", [
            "Oncology Consultation",
            "Mammogram",
            "Blood Test",
            "CT Scan",
            "MRI",
            "Surgery Consultation",
            "Follow-up Visit",
            "Chemotherapy",
            "Radiation Therapy",
            "Other"
        ])
        appt_doctor = st.text_input("Doctor/Clinic", placeholder="Dr. Smith - Cancer Center")
    
    with col2:
        appt_date = st.date_input("Appointment Date")
        appt_time = st.time_input("Appointment Time")

    appt_location = st.text_input("Location", placeholder="Hospital address or clinic name")
    appt_notes = st.text_area("Notes/Preparation", placeholder="Tests to bring, questions to ask...")

    if st.button("📝 Schedule Appointment"):
        new_appt = {
            "type": appt_type,
            "doctor": appt_doctor,
            "date": appt_date.strftime("%Y-%m-%d"),
            "time": appt_time.strftime("%H:%M"),
            "location": appt_location,
            "notes": appt_notes,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.appointments.append(new_appt)
        st.success(f"Scheduled {appt_type} for {appt_date} at {appt_time}")

    # Display upcoming appointments
    if st.session_state.appointments:
        st.markdown("### Upcoming Appointments")
        
        # Sort appointments by date
        sorted_appointments = sorted(st.session_state.appointments, key=lambda x: x['date'])
        
        for i, appt in enumerate(sorted_appointments):
            appt_datetime = datetime.strptime(f"{appt['date']} {appt['time']}", "%Y-%m-%d %H:%M")
            is_upcoming = appt_datetime >= datetime.now()
            
            status_emoji = "🔜" if is_upcoming else "✅"
            
            with st.expander(f"{status_emoji} {appt['type']} - {appt['date']} at {appt['time']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Doctor/Clinic:** {appt['doctor']}")
                    st.write(f"**Location:** {appt['location']}")
                with col2:
                    st.write(f"**Date & Time:** {appt['date']} at {appt['time']}")
                    st.write(f"**Notes:** {appt['notes']}")
                
                if st.button(f"🗑️ Cancel Appointment", key=f"cancel_appt_{i}"):
                    st.session_state.appointments.pop(i)
                    st.rerun()

        # Appointment reminders
        st.markdown("### 🔔 Upcoming This Week")
        today = datetime.now()
        week_from_now = today + pd.Timedelta(days=7)
        
        upcoming_this_week = []
        for appt in st.session_state.appointments:
            appt_date = datetime.strptime(appt['date'], "%Y-%m-%d")
            if today <= appt_date <= week_from_now:
                upcoming_this_week.append(appt)
        
        if upcoming_this_week:
            for appt in upcoming_this_week:
                st.warning(f"⚠️ {appt['type']} with {appt['doctor']} on {appt['date']} at {appt['time']}")
        else:
            st.info("No appointments scheduled for this week.")

    else:
        st.info("No appointments scheduled. Add your appointments above.")

# Educational Quiz Tab
with tab11:
    st.markdown("# 🎯 Breast Cancer Education Quiz")
    st.markdown("Test your knowledge about breast cancer and HER2!")

    questions = [
        {
            "question": "What percentage of breast cancers are HER2-positive?",
            "options": ["5-10%", "15-20%", "25-30%", "40-50%"],
            "correct": 1,
            "explanation": "About 15-20% of breast cancers are HER2-positive, meaning they have too much HER2 protein."
        },
        {
            "question": "What does HER2 stand for?",
            "options": ["Human Estrogen Receptor 2", "Human Epidermal Growth Factor Receptor 2", "Human Endocrine Receptor 2", "Human Emergency Response 2"],
            "correct": 1,
            "explanation": "HER2 stands for Human Epidermal Growth Factor Receptor 2, a protein that can promote cancer cell growth when overexpressed."
        },
        {
            "question": "At what age should women begin regular mammogram screening?",
            "options": ["30", "35", "40", "50"],
            "correct": 2,
            "explanation": "Most guidelines recommend starting annual mammograms at age 40, though this may vary based on family history and risk factors."
        },
        {
            "question": "Which of these is a targeted therapy for HER2-positive breast cancer?",
            "options": ["Tamoxifen", "Herceptin (Trastuzumab)", "Metformin", "Aspirin"],
            "correct": 1,
            "explanation": "Herceptin (Trastuzumab) is a targeted therapy specifically designed to treat HER2-positive breast cancers."
        },
        {
            "question": "What is the most common symptom of breast cancer?",
            "options": ["Breast pain", "A new lump or mass", "Nipple discharge", "Skin redness"],
            "correct": 1,
            "explanation": "A new lump or mass in the breast or underarm area is the most common symptom of breast cancer."
        }
    ]

    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        st.session_state.quiz_answers = []
        st.session_state.quiz_completed = False

    if not st.session_state.quiz_completed:
        current_q = st.session_state.current_question
        
        if current_q < len(questions):
            question_data = questions[current_q]
            
            st.markdown(f"### Question {current_q + 1} of {len(questions)}")
            st.markdown(f"**{question_data['question']}**")
            
            # Progress bar
            progress = (current_q) / len(questions)
            st.progress(progress)
            
            # Answer options
            answer = st.radio("Select your answer:", question_data['options'], key=f"q_{current_q}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⏭️ Next Question") and answer:
                    selected_index = question_data['options'].index(answer)
                    is_correct = selected_index == question_data['correct']
                    
                    st.session_state.quiz_answers.append({
                        'question': question_data['question'],
                        'selected': answer,
                        'correct': is_correct,
                        'explanation': question_data['explanation']
                    })
                    
                    if current_q < len(questions) - 1:
                        st.session_state.current_question += 1
                    else:
                        st.session_state.quiz_completed = True
                    
                    st.rerun()
            
            with col2:
                if st.button("🔄 Restart Quiz"):
                    st.session_state.current_question = 0
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_completed = False
                    st.rerun()
        
    else:
        # Quiz completed - show results
        st.markdown("# 🎉 Quiz Completed!")
        
        correct_answers = sum(1 for answer in st.session_state.quiz_answers if answer['correct'])
        total_questions = len(questions)
        score_percentage = (correct_answers / total_questions) * 100
        
        # Score display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{correct_answers}/{total_questions}")
        with col2:
            st.metric("Percentage", f"{score_percentage:.0f}%")
        with col3:
            if score_percentage >= 80:
                st.success("🌟 Excellent!")
            elif score_percentage >= 60:
                st.info("👍 Good job!")
            else:
                st.warning("📚 Keep learning!")
        
        # Detailed results
        st.markdown("### Detailed Results")
        for i, answer in enumerate(st.session_state.quiz_answers):
            with st.expander(f"Question {i+1}: {answer['question'][:50]}..."):
                status = "✅ Correct" if answer['correct'] else "❌ Incorrect"
                st.markdown(f"**{status}**")
                st.markdown(f"**Your answer:** {answer['selected']}")
                st.markdown(f"**Explanation:** {answer['explanation']}")
        
        # Educational resources based on score
        st.markdown("### 📚 Continue Learning")
        if score_percentage < 80:
            st.info("""
            **Recommended Resources:**
            - Review the 'About HER2 & Breast Cancer' tab
            - Check out the 'Symptoms Assessment' section
            - Visit the 'Support & Resources' tab for educational materials
            """)
        
        # Restart option
        if st.button("🔄 Take Quiz Again"):
            st.session_state.current_question = 0
            st.session_state.quiz_answers = []
            st.session_state.quiz_completed = False
            st.rerun()

st.markdown("---")
st.markdown("ℹ️ **Note:** This is a simulated educational tool based on image color analysis. Not a medical diagnosis. Always consult healthcare professionals for proper medical evaluation.")
