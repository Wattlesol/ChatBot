import mysql.connector
import os 


class DatabaseManager:
    def __init__(self):
        pass

    def get_connection(self):
        try:
            self.conn = mysql.connector.connect(pool_name="chatbot_pool")
        except mysql.connector.errors.PoolError:
            self.conn = mysql.connector.connect(
                host=os.getenv("MYSQL_HOST"),
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DB")
            )
        return self.conn

    def close_connection(self):
        if self.conn.is_connected():
            self.conn.close()

    def initialize_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                session_id VARCHAR(255) PRIMARY KEY,
                history BLOB NOT NULL,
                booked_slots TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_prompts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                system_prompt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS booked_appoints (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_fullname VARCHAR(255) NOT NULL,
                user_email VARCHAR(255) NOT NULL,
                appointment_start DATETIME NOT NULL,
                appointment_end DATETIME NOT NULL,
                duration INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sitemaps (
                id INT AUTO_INCREMENT PRIMARY KEY,
                sitemap_url TEXT NOT NULL,
                extracted_url TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sitemap_url, extracted_url)
            )
        """)

        conn.commit()

    def save_base_prompts(self, system_prompt):
        conn = self.get_connection()
        cursor = conn.cursor()
        query = """
            INSERT INTO system_prompts (system_prompt)
            VALUES (%s)
        """
        cursor.execute(query, (system_prompt,))  # Ensure the tuple is correctly passed
        conn.commit()

    def get_latest_base_prompts(self):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT system_prompt
            FROM system_prompts
            ORDER BY created_at DESC
            LIMIT 1
        """
        cursor.execute(query)
        return cursor.fetchone()
    
    def save_sitemap_to_db(self, sitemap_url, urls):
        """
        Save a single sitemap URL and its extracted URLs to the database.
        Remove any existing sitemaps before saving the new one.
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Delete all existing sitemap entries
        cursor.execute("DELETE FROM sitemaps")
        conn.commit()

        # Insert the new sitemap URL and its URLs
        query = """
            INSERT INTO sitemaps (sitemap_url, extracted_url)
            VALUES (%s, %s)
        """
        for url in urls:
            cursor.execute(query, (sitemap_url, url))
        conn.commit()
        