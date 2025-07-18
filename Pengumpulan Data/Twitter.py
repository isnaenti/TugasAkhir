import os

# Ganti ini dengan token Twitter API milik Anda
twitter_auth_token = "ade"

# Daftar keyword saham
keywords = ["$ISAT"]

# Loop untuk crawl masing-masing keyword
for keyword in keywords:
    filename = f"{keyword.lower()}_tweetISAT.csv"
    search_keyword = f"{keyword} since:2019-01-01 until:2025-07-09 lang:id"
    limit = 5000

    # Format perintah tweet-harvest
    command = f'npx -y tweet-harvest@2.6.1 -o "{filename}" -s "{search_keyword}" --tab "LATEST" -l {limit} --token "{twitter_auth_token}"'
    
    # Jalankan perintah
    os.system(command)

    print(f"✅ Data untuk {keyword} disimpan di {filename}")
