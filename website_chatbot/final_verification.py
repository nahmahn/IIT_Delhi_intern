import asyncio
from rag import process_query
import json

async def run_tests():
    test_queries = [
        "What are the main clusters for Negamam cotton sarees?", # 1. Image Leakage/Specific Answer
        "What is the total carbon footprint of a Baluchari silk saree?", # 2. Pie Chart Caption/Data
        "How much does wastewater contribute to Baluchari emissions?", # 3. Specific Chart Data
        "Compare the weaving techniques of Phulkari and Muslin." # 4. Comparison/Concise Response
    ]
    
    print("--- STARTING FINAL VERIFICATION ---")
    with open("verification_report.txt", "w", encoding="utf-8") as rf:
        for q in test_queries:
            print(f"\nQUERY: {q}")
            rf.write(f"\nQUERY: {q}\n")
            
            # process_query returns a dict
            result = process_query(q)
            answer = result["answer"]
            images = result["images"]
            
            # Print with ASCII safety for console, full UTF-8 for file
            print(f"ANSWER: {answer.encode('ascii', 'ignore').decode('ascii')[:100]}...")
            rf.write(f"ANSWER: {answer}\n")
            
            print(f"IMAGES FOUND: {len(images)}")
            rf.write(f"IMAGES FOUND: {len(images)}\n")
            for img in images:
                img_log = f"  - {img['url']} (Caption: {img['description']}, Project: {img['project']})"
                print(img_log.encode('ascii', 'ignore').decode('ascii'))
                rf.write(img_log + "\n")
            
            print("-" * 50)
            rf.write("-" * 50 + "\n")
    
    print("\nVerification process complete. Full report in verification_report.txt")

if __name__ == "__main__":
    asyncio.run(run_tests())
