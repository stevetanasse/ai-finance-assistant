Always use explicit UTF-8 encoding for all file I/O operations in this project. Apply the following rules consistently across every file you create or modify:

1. Python open() calls
   Always pass encoding="utf-8" explicitly:
     open(path, "r", encoding="utf-8")
     open(path, "w", encoding="utf-8")
   Never rely on the default locale encoding — it is cp1252 on Windows
   and will silently corrupt or fail to read Unicode content.

2. pathlib read_text() / write_text()
   Always pass encoding="utf-8":
     path.read_text(encoding="utf-8")
     path.write_text(content, encoding="utf-8")

3. json.dump() / json.dumps()
   Pair with encoding="utf-8" on the file handle and prefer
   ensure_ascii=False so Unicode characters are stored as-is rather
   than as \uXXXX escape sequences:
     with open(path, "w", encoding="utf-8") as f:
         json.dump(data, f, ensure_ascii=False)

4. JSONL files
   Apply the same rules — open with encoding="utf-8" and use
   ensure_ascii=False when serialising each line:
     with open(path, "w", encoding="utf-8") as f:
         for record in records:
             f.write(json.dumps(record, ensure_ascii=False) + "\n")

5. Test assertions
   Any test that reads a file must also specify encoding="utf-8":
     file.read_text(encoding="utf-8")
   Do not call read_text() or open() without an explicit encoding
   argument anywhere in the test suite.

# Rationale: web-scraped content and JSON data routinely contain
# characters outside the ASCII range. Relying on the platform default
# encoding causes UnicodeDecodeError on Windows (cp1252) and silent
# data corruption on other systems. Explicit UTF-8 is the correct
# encoding for all JSON, JSONL, and general text files.
