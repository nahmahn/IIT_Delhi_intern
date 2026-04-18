from langchain_text_splitters import CharacterTextSplitter
import json

def chunk_record(record, source_type):
    text = record.get("transcript") or record.get("content", "")
    if not text:
        return []
    
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1600,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = splitter.split_text(text)
    
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "course_id":    record["course_id"],
            "course_title": record["course_title"],
            "professor":    record["professor"],
            "institute":    record["institute"],
            "lecture_name": record["lecture_name"],
            "source_type":  source_type,
            "source_url":   record.get("youtube_url") or record.get("lecture_url", ""),
            "chunk_index":  i,
            "chunk_total":  len(chunks),
            "chunk_text":   chunk,
            "char_count":   len(chunk)
        })
    
    return results

all_chunks = []

# ── HTML lectures only for now ─────────────────────────────
with open("textile_html_lectures.json") as f:
    html_lectures = json.load(f)

for record in html_lectures:
    all_chunks.extend(chunk_record(record, "html"))

print(f"HTML chunks:    {len(all_chunks)}")
print(f"Total words:    {sum(c['char_count'] for c in all_chunks) // 5:,}")
print(f"Avg chunk size: {sum(c['char_count'] for c in all_chunks) // len(all_chunks)} chars")

with open("textile_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Saved → textile_chunks.json")
print(f"\nTomorrow — run this to add video transcripts:")
print(f"  python chunk_add_videos.py")