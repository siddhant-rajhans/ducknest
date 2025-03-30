# 🏠 DuckNest  

### Solving Student Housing Challenges at Stevens  

## 📌 Problem Statement  
Finding accommodation can be a significant challenge for new international and national students at **Stevens Institute of Technology**. Existing rental platforms often involve brokers, additional fees, and unreliable listings. **DuckNest** aims to solve this by enabling senior students to list available accommodations directly—eliminating middlemen and making the process transparent and affordable.  

## 🚀 Solution  
DuckNest is a **community-driven housing platform** built specifically for Stevens students. With our **Flutter** mobile app and a **Python-based backend**, students can seamlessly list, search, and secure housing without broker intervention.  

## 🛠️ Tech Stack  
- **Frontend:** Flutter  
- **Backend:** Python (Flask)  
- **Database:** Supabase (PostgreSQL)

## 🎯 Features  
✅ **Student-to-Student Listings:** Seniors list available properties for juniors and new students.  
✅ **No Broker Involvement:** Completely transparent process with direct student communication.  
✅ **Secure Authentication:** Only verified Stevens students can access the platform.  
✅ **Advanced Search & Filters:** Find accommodation based on price, location, and amenities.  
✅ **Chat & Inquiry System:** Contact property listers directly via the app.  

## 🔧 Setup Instructions  

### 📌 Backend (Python)  
1. Clone the repository:  
   ```bash
   git clone https://github.com/siddhant-rajhans/ducknest.git  
   cd ducknest/server 
   ```  
2. Create a virtual environment & install dependencies:  
   ```bash
   python -m venv venv  
   # for mac
   source venv/bin/activate  
   # On Windows, use 
   venv\Scripts\activate  
   pip install -r requirements.txt  
   ```  
3. Set up environment variables (Refer to `.env.example`)  
4. Run the server:  
   ```bash
   cd server
   python main.py  
   ```  

### 📌 Frontend (Flutter)  
1. Move to the Flutter project directory:  
   ```bash
   cd ../frontend/app4  
   ```  
2. Install dependencies:  
   ```bash
   flutter pub get  
   ```  
3. Run the app:  
   ```bash
   flutter run  
   ```  

## 📅 Future Enhancements  
🔹 Reviews & Ratings for listings  
🔹 AI-based property recommendations  
🔹 Map-based property search  
🔹 Rent splitting & roommate finder  

## 🤝 Contributors  
👨‍💻 **Siddhant Rajhans** & **Anirudh Pande** & **Prakhar Tripathi** & **Gunjan Rawat**
