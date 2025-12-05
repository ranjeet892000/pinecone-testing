

# CODE FOR PROCESSING 100 ROWS OF METADATA.JSONL

# import os
# import json
# from pinecone import Pinecone
# from pathlib import Path

# from dotenv import load_dotenv
# load_dotenv()

# # Import utility modules
# from image_utils import initialize_s3_client
# from background_utils import (
#     closest_match_with_removing_background,
#     closest_match_without_removing_background
# )
# from extract_metadata import process_jsonl_file

# # Initialize Pinecone
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "luxury-v2"
# pinecone_index = pc.Index(index_name)

# # Initialize S3 client
# s3_client = initialize_s3_client()



# def extract_session_ids(id_list):
#     """
#     Extract session IDs from a list of IDs.
    
#     Args:
#         id_list: List of IDs in format "session_uuid.macro.region_id.index"
    
#     Returns:
#         List of unique session IDs (first part before the first dot)
#     """
#     result = [s.split('.')[0] for s in id_list]
#     return result[:5]


# def process_100_rows():
#     """
#     Get 100 rows from metadata.jsonl (with region_id='camera') and call both 
#     closest_match_with_removing_background and closest_match_without_removing_background for each.
#     Returns a JSON structure mapping each row ID to arrays of result IDs from both methods.
#     """
#     print("📂 Loading 100 records from metadata.jsonl (region_id='camera')...")
#     records = process_jsonl_file('metadata.jsonl', max_records=100, region_id_filter='camera')
#     print(f"✅ Loaded {len(records)} records\n")
    
#     results_dict = {}
    
#     for idx, record in enumerate(records, 1):
#         print(f"Processing record {idx}/{len(records)}")
        
#         # Construct ID from available data
#         # Format: {session_uuid}.macro.{region_id}.0
#         session_uuid = record.get('session_uuid', '')
#         region_id = record.get('region_id', '')
#         constructed_id = f"{session_uuid}.macro.camera.0"
        
#         s3_key = record.get('s3_key')
#         brand_id = record.get('brand_id')
        
#         if s3_key:
#             try:
#                 # Call both functions
#                 print(f"\n🔍 Running closest_match_with_removing_background...")
#                 results_with_removing_bg = closest_match_with_removing_background(
#                     pinecone_index=pinecone_index,
#                     s3_client=s3_client,
#                     id=constructed_id,
#                     s3_key=s3_key,
#                     brand_id=brand_id
#                 )
                
#                 print(f"\n🔍 Running closest_match_without_removing_background...")
#                 results_without_removing_bg = closest_match_without_removing_background(
#                     pinecone_index=pinecone_index,
#                     id=constructed_id
#                 )
                
#                 # Store results with separate sections for with_background and without_background
#                 results_dict[constructed_id] = {
#                     "with_background": results_with_removing_bg,
#                     "without_background": results_without_removing_bg
#                 }
                
#                 # print(f"\n✅ Record {idx} processed: {len(results_with_removing_bg)} results (with bg) + {len(results_without_removing_bg)} results (without bg)")
                
#             except Exception as e:
#                 print(f"❌ Error processing record {idx}: {e}")
#                 # Store empty lists for failed records
#                 results_dict[constructed_id] = {
#                     "with_background": [],
#                     "without_background": []
#                 }
#                 continue
#         else:
#             print(f"⚠️  Skipping record {idx}: No s3_key found")
#             results_dict[constructed_id] = {
#                 "with_background": [],
#                 "without_background": []
#             }
    
#     print(f"\n✅ Finished processing {len(records)} records")
    
    
#     # Also save to file
#     output_file = "results.json"
#     with open(output_file, 'w') as f:
#         json.dump(results_dict, f, indent=2)
#     print(f"\n💾 Results saved to {output_file}")
    
#     return results_dict


# if __name__ == '__main__':
#     # process_100_rows()
#     id = "005aebd9-c260-487b-99de-63eab6aae207.macro.camera.0"
#     results_camera = closest_match_without_removing_background(pinecone_index, id,"camera")
#     session_ids_camera = extract_session_ids(results_camera)


#     id = "005aebd9-c260-487b-99de-63eab6aae207.macro.inner_logo.0"
#     results_inner_logo = closest_match_without_removing_background(pinecone_index, id,"inner_logo",session_ids_camera)
    
    
#     print(f"Camera results: {results_camera}")
#     print(f"Session IDs from camera: {session_ids_camera}")
#     print(f"Inner logo results: {results_inner_logo}")
#     print(f"Session IDs from inner logo: {extract_session_ids(results_inner_logo)}")
#     # print(f"Session IDs from inner_logo: {session_ids_inner_logo}")



# // CODE FOR EXACT MATCH PIPELINE USING ORB AND COLOR HISTOGRAM SIMILARITY?

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
