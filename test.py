import os
from pinecone import Pinecone

from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pranav-test-prod"

# print(pc.list_indexes())
pinecone_index = pc.Index(index_name)

# To get the unique host for an index, 
# see https://docs.pinecone.io/guides/manage-data/target-an-index
# index = pc.Index(host="INDEX_HOST")

response=pinecone_index.fetch(ids=["87a46d8c-45e7-4fe8-9185-f0a509c5d421.macro.camera.0"], namespace="")
vector = response.vectors["87a46d8c-45e7-4fe8-9185-f0a509c5d421.macro.camera.0"]

# Access the metadata
metadata = vector.metadata

# Get the s3_key
s3_key = metadata.get("s3_key")
print(s3_key)



import boto3

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

bucket_name = "entrupy-app-db"
object_key = "authentications/87a46d8c-45e7-4fe8-9185-f0a509c5d421/01020005-f560-7c48-ac9e-6806000001dc/kv1/2.13.0/device_alias/camera0/mime_type/image/jpeg/region_id/camera/overlay_key/null/file_id/0/content.jpg"

s3.download_file(bucket_name, object_key, "downloaded_image.jpg")
print("Downloaded!")





# import os
# import io
# import numpy as np
# import requests
# from PIL import Image
# from dotenv import load_dotenv
# from pinecone import Pinecone
# import boto3
# from botocore.exceptions import ClientError
# import imagehash
# import cv2
# from skimage.metrics import structural_similarity as ssim

# load_dotenv()

# # Pinecone init
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "pranav-test-prod"
# index = pc.Index(index_name)

# # boto3 for presign
# s3 = boto3.client("s3",
#                   aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#                   aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#                   region_name=os.getenv("AWS_REGION"))


# def make_presigned_url(bucket, key, expires_in=300):
#     try:
#         return s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in)
#     except ClientError as e:
#         print("Presign error:", e)
#         return None


# def load_image_from_url(url):
#     resp = requests.get(url, timeout=10)
#     resp.raise_for_status()
#     return Image.open(io.BytesIO(resp.content)).convert("RGB")


# def phash_distance(img1: Image.Image, img2: Image.Image):
#     return imagehash.phash(img1) - imagehash.phash(img2)  # Hamming distance (int)


# def compute_ssim(img1: Image.Image, img2: Image.Image):
#     # convert to grayscale numpy arrays, resize to same size
#     a = np.array(img1.convert("L"))
#     b = np.array(img2.convert("L"))
#     # resize b to a's shape if needed
#     if a.shape != b.shape:
#         b = cv2.resize(b, (a.shape[1], a.shape[0]))
#     score, _ = ssim(a, b, full=True)
#     return float(score)


# def orb_match_count(img1: Image.Image, img2: Image.Image):
#     a = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
#     b = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
#     # ORB detector
#     orb = cv2.ORB_create(1000)
#     kp1, des1 = orb.detectAndCompute(a, None)
#     kp2, des2 = orb.detectAndCompute(b, None)
#     if des1 is None or des2 is None:
#         return 0
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     # sort by distance
#     matches = sorted(matches, key=lambda x: x.distance)
#     # return number of good matches (you can threshold by distance)
#     good = [m for m in matches if m.distance < 60]
#     return len(good)


# def exact_match_pipeline(query_image_path=None, query_id=None, top_k=10):
#     # If you have an ID, query by id else embed and query by vector (not shown)
#     query_params = {
#         "id": query_id,
#         "top_k": top_k,
#         "include_metadata": True
#     }
#     results = index.query(**query_params)
#     matches = results.matches if hasattr(results, "matches") else results.get("matches", [])

#     # load query image
#     query_img = Image.open(query_image_path).convert("RGB") if query_image_path else None

#     scored = []
#     for m in matches:
#         meta = m.get("metadata", {})
#         bucket = meta.get("s3_bucket")
#         key = meta.get("s3_key")
#         if not (bucket and key):
#             continue
#         url = make_presigned_url(bucket, key, expires_in=300)
#         if not url:
#             continue
#         try:
#             candidate_img = load_image_from_url(url)
#         except Exception as e:
#             print("Failed to load candidate:", e)
#             continue

#         # 1) quick phash check
#         ph = phash_distance(query_img, candidate_img)

#         # 2) SSIM
#         s = compute_ssim(query_img, candidate_img)

#         # 3) ORB matches
#         orb_matches = orb_match_count(query_img, candidate_img)

#         scored.append({
#             "id": m.get("id"),
#             "score": m.get("score"),
#             "phash_hamming": int(ph),
#             "ssim": s,
#             "orb_matches": orb_matches,
#             "metadata": meta,
#             "url": url
#         })

#     # Sort candidates: prefer phash small, ssim high, orb_matches high
#     scored.sort(key=lambda x: (x["phash_hamming"], -x["ssim"], -x["orb_matches"]))

#     # Decide thresholds for exact match:
#     # - phash_hamming <= 6
#     # - ssim >= 0.85
#     # - orb_matches >= 30
#     # tweak these values based on experiments on your dataset.
#     exact_candidates = [c for c in scored if (c["phash_hamming"] <= 6 and c["ssim"] >= 0.85) or c["orb_matches"] >= 40]

#     return scored, exact_candidates


# # Example usage:
# scored, exact = exact_match_pipeline(query_image_path="content.jpg", query_id="87a46d8c-45e7-4fe8-9185-f0a509c5d421.macro.camera.0", top_k=100)
# print("Top 10 candidates (summary):")
# for s in scored[:10]:
#     print(s["id"], "ph:", s["phash_hamming"], "ssim:", round(s["ssim"], 3), "orb:", s["orb_matches"])
# print("Exact candidates:", [c['id'] for c in exact])







# import os
# import io
# import numpy as np
# import requests
# from PIL import Image
# from dotenv import load_dotenv
# from pinecone import Pinecone
# import boto3
# from botocore.exceptions import ClientError
# import cv2

# load_dotenv()

# # Pinecone init
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "pranav-test-prod"
# index = pc.Index(index_name)

# # boto3 for presign
# s3 = boto3.client(
#     "s3",
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#     region_name=os.getenv("AWS_REGION")
# )


# def make_presigned_url(bucket, key, expires_in=300):
#     try:
#         return s3.generate_presigned_url("get_object",
#                                          Params={"Bucket": bucket, "Key": key},
#                                          ExpiresIn=expires_in)
#     except ClientError as e:
#         print("Presign error:", e)
#         return None


# def load_image_from_url(url):
#     resp = requests.get(url, timeout=10)
#     resp.raise_for_status()
#     return Image.open(io.BytesIO(resp.content)).convert("RGB")


# # ---------------------------
# # ORB FEATURE MATCHING
# # ---------------------------
# def orb_match_count(img1: Image.Image, img2: Image.Image):
#     a = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
#     b = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

#     orb = cv2.ORB_create(1000)
#     kp1, des1 = orb.detectAndCompute(a, None)
#     kp2, des2 = orb.detectAndCompute(b, None)

#     if des1 is None or des2 is None:
#         return 0

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)

#     matches = sorted(matches, key=lambda x: x.distance)

#     good = [m for m in matches if m.distance < 60]
#     return len(good)


# # ---------------------------
# # COLOR HISTOGRAM SIMILARITY
# # ---------------------------
# def color_hist_similarity(img1: Image.Image, img2: Image.Image):
#     """Return similarity score between 0 and 1."""
#     a = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2HSV)
#     b = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2HSV)

#     histA = cv2.calcHist([a], [0, 1], None, [50, 50], [0, 180, 0, 256])
#     histB = cv2.calcHist([b], [0, 1], None, [50, 50], [0, 180, 0, 256])

#     cv2.normalize(histA, histA)
#     cv2.normalize(histB, histB)

#     score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
#     return float(score)


# def exact_match_pipeline(query_image_path=None, query_id=None, top_k=10):
#     query_params = {
#         "id": query_id,
#         "top_k": top_k,
#         "include_metadata": True
#     }

#     results = index.query(**query_params)
#     matches = results.matches if hasattr(results, "matches") else results.get("matches", [])

#     query_img = Image.open(query_image_path).convert("RGB")

#     scored = []

#     for m in matches:
#         meta = m.get("metadata", {})
#         bucket = meta.get("s3_bucket")
#         key = meta.get("s3_key")

#         if not (bucket and key):
#             continue

#         url = make_presigned_url(bucket, key)
#         if not url:
#             continue

#         try:
#             candidate_img = load_image_from_url(url)
#         except Exception as e:
#             print("Failed loading:", e)
#             continue

#         # ORB matching
#         orb_matches = orb_match_count(query_img, candidate_img)

#         # Color similarity
#         color_sim = color_hist_similarity(query_img, candidate_img)

#         scored.append({
#             "id": m.get("id"),
#             "pinecone_score": m.get("score"),
#             "orb_matches": orb_matches,
#             "color_similarity": color_sim,
#             "url": url,
#             "metadata": meta
#         })

#     # Sort by: more ORB matches + higher color similarity
#     scored.sort(key=lambda x: (-x["orb_matches"], -x["color_similarity"]))

#     # USE STRICT FILTERS FOR EXACT MATCH
#     exact_candidates = [
#         c for c in scored
#         if c["orb_matches"] >= 40 and c["color_similarity"] >= 0.7
#     ]

#     return scored, exact_candidates


# # Run
# scored, exact = exact_match_pipeline(
#     query_image_path="content.jpg",
#     query_id="87a46d8c-45e7-4fe8-9185-f0a509c5d421.macro.camera.0",
#     top_k=10
# )

# print("Top 10:")
# for s in scored[:10]:
#     print(s["id"], "ORB:", s["orb_matches"], "COLOR:", round(s["color_similarity"], 3),"url:", s["url"])

# print("Exact matches:", [c["id"] for c in exact])
