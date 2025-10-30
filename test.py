# s = "Home - Heinz History Center Play Video Pause Video Always Free for Kids! Plan Your Visit Get to Know Our Family of Museums Heinz History Center 10 AM - 5 PM Western Pennsylvania Sports Museum 10 AM - 5 PM Fort Pitt Museum 10 AM - 5 PM Meadowcroft Rockshelter and Historic Village 10 AM - 4 PM Pittsburgh’s Hidden History Exhibit Heinz History Center Wander through a whimsical world of wonders inside our dynamic new exhibition, featuring a yinzsplosion of rarely and never-before-seen objects that tell stories from Pittsburgh’s past. Uncover More Haunted History Center Halloween Events Heinz History Center Step inside the History Center for a spooktacular day of tricks, treats, and ghoulishly grand fun on Oct. 26! Learn More What's On: Events Search our calendar of upcoming events hosted by our family of museums! View All Events October 11 Fort Pitt Museum – 1:00 PM Speaker Saturday: Overshadowed by These Occult Influences Join the Fort Pitt Museum to learn about the hidden beliefs that lingered in the Ohio Valley. October 16 Heinz History Center – 6:30 PM Hidden History Trivia Night It’s a dark and stormy trivia night in Pittsburgh…with a spooky twist! Tickets Required October 18 Meadowcroft Rockshelter and Historic Village – 10:00 AM Archaeology Day Dig into the past at Meadowcroft’s Archaeology Day! View All Events Preserving Pittsburgh’s Memories Detre Library & Archives Heinz History Center Research family history, explore historic images, and search thousands of documents in the Detre Library & Archives at the History Center. Open Wednesday through Saturday and free to all visitors. Research & Explore More than a ketchup museum. The Heinz History Center is Pittsburgh’s people museum. We share the inspiring stories of Western Pennsylvania’s people who have helped change the course of American history. See for yourself. Explore Exhibits Kids & Families Build bridges in the interactive Discovery Place or explore the Neighborhood of Make–Believe. Unique Pittsburgh Gifts From exclusive Heinz merch to the Mister Rogers kindness collection, find the perfect Pittsburgh gift at the Museum Shop. Smithsonian Treasures Discover Smithsonian artifacts at the History Center and learn more about the museum’s Smithsonian affiliation. Explore Our Collections Thousands of artifacts and historic images, at your fingertips. Kids & Families Build bridges in the interactive Discovery Place or explore the Neighborhood of Make–Believe. Unique Pittsburgh Gifts From exclusive Heinz merch to the Mister Rogers kindness collection, find the perfect Pittsburgh gift at the Museum Shop. Smithsonian Treasures Discover Smithsonian artifacts at the History Center and learn more about the museum’s Smithsonian affiliation. Explore Our Collections Thousands of artifacts and historic images, at your fingertips. Site Menu Search Submit Close"

# def chunk_text(text, chunk_size=300, overlap=20):
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i:i + chunk_size])
#         if len(chunk) > 0:
#             chunks.append(chunk)
#     return chunks

# chunks = chunk_text(s)

# print(chunks)

# import numpy as np

# data = np.load("index/embeddings.npy")
# print(data.shape)
# print(data)

from dense_retrieve import QUESTIONS_PATH

with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

print(len(questions))