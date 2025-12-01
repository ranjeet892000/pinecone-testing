import os
import streamlit as st
from pinecone import Pinecone
import pandas as pd
from typing import List
import boto3
from botocore.exceptions import ClientError



from dotenv import load_dotenv
load_dotenv()
# Configure Streamlit page
st.set_page_config(
    page_title="Luxury Brand Image Search",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .combined-score-header {
        background: linear-gradient(90deg, #e7f3e7 0%, #c8e6c9 100%);
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
        font-weight: bold;
        color: #2e7d32;
        font-size: 1.2em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .region-header {
        background-color: #f0f2f6;
        padding: 12px 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1.1em;
    }
    .image-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px;
        background: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .image-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-top: 5px;
    }
    .rank-badge {
        position: absolute;
        top: 5px;
        left: 5px;
        background: #ff4b4b;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        z-index: 10;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc

# Initialize S3 client
@st.cache_resource
def get_s3_client():
    """Initialize and cache S3 client"""
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not aws_access_key or not aws_secret_key:
        return None

    try:
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
    except Exception as e:
        st.error(f"Error initializing S3 client: {e}")
        return None

def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for an S3 object"""
    try:
        s3_client = get_s3_client()
        if s3_client is None:
            return None
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        return url
    except ClientError:
        return None
    except Exception:
        return None

def get_image_url_from_match(match) -> str:
    """Extract S3 info and generate presigned URL for a match"""
    if not match:
        return None

    # Convert ScoredVector to dict if needed
    match_dict = match_to_dict(match)
    if not match_dict:
        return None

    metadata = match_dict.get('metadata', {})
    if not isinstance(metadata, dict):
        return None

    bucket = metadata.get('s3_bucket')
    key = metadata.get('s3_key')

    if bucket and key:
        return generate_presigned_url(bucket, key)
    return None

def fetch_all_id(brand_id, session_id, region_ids, top_k=50, index=None):
    """
    Fetch matches for all regions.
    """
    if index is None:
        pc = init_pinecone()
        index = pc.Index("luxury-v2")

    matches = {}
    for region_id in region_ids:
        query_params = {
            "id": f"{session_id}.{region_id}.0",
            "top_k": top_k,
            "include_metadata": True,
            "filter": {
                "region_id": region_id.split(".")[1],
                "brand_id": brand_id
            }
        }
        results = index.query(**query_params)
        matches[region_id] = results['matches']
    return matches

def fetch_common_session_id(matches):
    common_session_ids = set()
    for region_id, region_matches in matches.items():
        for match in region_matches:
            if not match:
                continue
            match_dict = match_to_dict(match)
            if match_dict and 'id' in match_dict:
                try:
                    match_id = match_dict['id']
                    if isinstance(match_id, str):
                        common_session_ids.add(match_id.split(".")[0])
                except (AttributeError, KeyError, IndexError, TypeError):
                    pass
    return list(common_session_ids)

def match_to_dict(match):
    """Convert ScoredVector object to dictionary for easier handling"""
    if isinstance(match, dict):
        return match

    if not match:
        return None

    # Handle ScoredVector objects from Pinecone SDK
    try:
        # Try accessing as attributes first (ScoredVector objects)
        match_dict = {}

        # Get id
        if hasattr(match, 'id'):
            match_dict['id'] = match.id
        elif hasattr(match, '__getitem__'):
            try:
                match_dict['id'] = match['id']
            except (KeyError, TypeError):
                pass

        # Get score
        if hasattr(match, 'score'):
            match_dict['score'] = match.score
        elif hasattr(match, '__getitem__'):
            try:
                match_dict['score'] = match.get('score') if hasattr(match, 'get') else match['score']
            except (KeyError, TypeError, AttributeError):
                match_dict['score'] = None

        # Get metadata
        if hasattr(match, 'metadata'):
            match_dict['metadata'] = match.metadata or {}
        elif hasattr(match, '__getitem__'):
            try:
                match_dict['metadata'] = match.get('metadata', {}) if hasattr(match, 'get') else match.get('metadata', {})
            except (KeyError, TypeError, AttributeError):
                match_dict['metadata'] = {}

        # Get values if available
        if hasattr(match, 'values'):
            match_dict['values'] = match.values

        return match_dict if match_dict else None
    except Exception:
        # Last resort: try dict conversion
        try:
            if hasattr(match, '__dict__'):
                return match.__dict__
            elif hasattr(match, '__iter__') and not isinstance(match, str):
                return dict(match)
        except Exception:
            pass
        return None

def rerank_session_ids_with_common_score(matches, query_session_id=None):
    """
    Combine scores from all regions for each session_uuid and return sorted by combined score.

    Args:
        matches: Dictionary of region_id -> list of matches
        query_session_id: Session ID to exclude from results (optional)
    """
    session_scores = {}

    for region_id, region_matches in matches.items():
        for match in region_matches:
            if not match:
                continue

            # Convert ScoredVector to dict if needed
            match_dict = match_to_dict(match)
            if not match_dict or 'id' not in match_dict:
                continue

            # Extract session UUID safely
            try:
                match_id = match_dict['id']
                if not isinstance(match_id, str):
                    continue
                session_uuid = match_id.split(".")[0]
            except (AttributeError, KeyError, IndexError, TypeError):
                continue

            # # Task 1: Exclude query session from results
            # if query_session_id and session_uuid == query_session_id:
            #     continue

            score = match_dict.get('score', 0.0)
            metadata = match_dict.get('metadata', {})

            if session_uuid not in session_scores:
                session_scores[session_uuid] = {
                    'scores': [],
                    'region_scores': {},
                    'metadata': metadata
                }

            session_scores[session_uuid]['scores'].append(score)
            session_scores[session_uuid]['region_scores'][region_id] = score

    ranked_sessions = []
    for session_uuid, data in session_scores.items():
        combined_score = sum(data['scores'])
        ranked_sessions.append({
            'session_uuid': session_uuid,
            'combined_score': combined_score,
            'average_score': combined_score / len(data['scores']),
            'region_scores': data['region_scores'],
            'num_regions': len(data['scores']),
            'metadata': data['metadata']
        })

    ranked_sessions.sort(key=lambda x: x['combined_score'], reverse=True)
    return ranked_sessions

def display_image_row(matches: List[dict], items_per_row: int = 5, show_rank: bool = True):
    """Display images in a horizontal scrollable row using Streamlit columns"""
    if not matches:
        return

    # Create rows of columns
    num_items = len(matches)
    num_rows = (num_items + items_per_row - 1) // items_per_row

    for row_idx in range(num_rows):
        cols = st.columns(items_per_row)
        start_idx = row_idx * items_per_row
        end_idx = min(start_idx + items_per_row, num_items)

        for col_idx, match_idx in enumerate(range(start_idx, end_idx)):
            match = matches[match_idx]

            # Convert ScoredVector to dict if needed
            match_dict = match_to_dict(match)
            if not match_dict:
                with cols[col_idx]:
                    st.warning("Invalid match data")
                    continue

            if 'id' not in match_dict:
                with cols[col_idx]:
                    st.warning("Match missing ID field")
                    continue

            with cols[col_idx]:
                # Get image URL from S3
                image_url = get_image_url_from_match(match)
                session_uuid = match_dict['id'].split(".")[0] if match_dict.get('id') else "unknown"
                score = match_dict.get('score', 0.0)  # Use default if score missing
                rank = match_idx + 1 if show_rank else None

                # Rank badge
                # if rank:
                #     st.markdown(f'<div style="background: #ff4b4b; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; font-weight: bold; text-align: center; margin-bottom: 5px;">#{rank}</div>', unsafe_allow_html=True)

                # Display image
                if image_url:
                    try:
                        st.image(
                            image_url,
                            caption=f"Score: {rank}({score:.4f}) Session: `{session_uuid[:12]}...",
                            use_container_width=True
                        )
                    except Exception:
                        st.error("Error loading image")
                        st.text(f"URL: {image_url[:50]}...")
                else:
                    st.warning("No image available")
                    metadata = match.get('metadata', {})
                    st.text(f"Bucket: {metadata.get('s3_bucket', 'N/A')}")
                    st.text(f"Key: {metadata.get('s3_key', 'N/A')[:30]}...")

# Streamlit App
def main():
    st.title("🔍 Luxury Brand Image Search Dashboard")
    st.markdown("Search and compare images across different regions with similarity scores")

    # Check AWS credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key or not aws_secret_key:
        st.warning("⚠️ AWS credentials not set! Images may not load. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Search Parameters")

        session_id = st.text_input(
            "Session ID",
            value="36033bf0-f0da-438c-b0b4-8bd46fc36306",
            help="Enter the session UUID"
        )

        brand_id = st.text_input(
            "Brand ID",
            value="louis_vuitton",
            help="Enter the brand identifier"
        )

        region_ids_input = st.text_area(
            "Region IDs (one per line)",
            value="macro.camera\nmicro.inner_logo\nmacro.inner_logo",
            help="Enter region IDs, one per line"
        )

        top_k = st.slider(
            "Top K Results",
            min_value=10,
            max_value=100,
            value=10,
            step=10,
            help="Number of top results to return per region"
        )

        search_button = st.button("🔎 Search", type="primary", use_container_width=True)

    # Main content area
    if search_button:
        region_ids = [rid.strip() for rid in region_ids_input.split('\n') if rid.strip()]

        if not region_ids:
            st.error("Please enter at least one region ID")
            return

        with st.spinner("Fetching results from Pinecone..."):
            try:
                pc = init_pinecone()
                pinecone_index = pc.Index("luxury-v2")
                results = fetch_all_id(brand_id, session_id, region_ids, top_k, pinecone_index)

                # Debug: Check match structure before filtering
                total_before_filter = sum(len(matches) for matches in results.values())

                ranked = rerank_session_ids_with_common_score(results, query_session_id=session_id)

                # Debug: Count sessions before filtering
                all_session_ids = set()
                for region_id, region_matches in results.items():
                    for match in region_matches:
                        if not match:
                            continue
                        match_dict = match_to_dict(match)
                        if match_dict and 'id' in match_dict:
                            try:
                                match_id = match_dict['id']
                                if isinstance(match_id, str):
                                    session_uuid = match_id.split(".")[0]
                                    all_session_ids.add(session_uuid)
                            except (AttributeError, KeyError, IndexError, TypeError):
                                pass

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Regions", len(region_ids))
                with col2:
                    st.metric("Unique Sessions", len(ranked))
                    if len(ranked) == 0 and len(all_session_ids) > 0:
                        st.caption(f"({len(all_session_ids)} before filter)")
                with col3:
                    if ranked:
                        st.metric("Best Combined Score", f"{ranked[0]['combined_score']:.4f}")
                with col4:
                    st.metric("Total Matches", total_before_filter)

                st.divider()

                # Debug info if no ranked sessions
                if len(ranked) == 0:
                    with st.expander("🔍 Debug: Why no ranked sessions?"):
                        st.write(f"Query Session ID: `{session_id}`")
                        st.write(f"Total matches across all regions: {total_before_filter}")
                        st.write(f"Unique session IDs found (before filter): {len(all_session_ids)}")

                        # Show session ID distribution
                        session_counts = {}
                        for region_id, region_matches in results.items():
                            for match in region_matches:
                                if not match:
                                    continue
                                match_dict = match_to_dict(match)
                                if match_dict and 'id' in match_dict:
                                    try:
                                        match_id = match_dict['id']
                                        if isinstance(match_id, str):
                                            session_uuid = match_id.split(".")[0]
                                            session_counts[session_uuid] = session_counts.get(session_uuid, 0) + 1
                                    except (AttributeError, KeyError, IndexError, TypeError):
                                        pass

                        if session_counts:
                            st.write("**Session ID distribution:**")
                            sorted_sessions = sorted(session_counts.items(), key=lambda x: x[1], reverse=True)
                            for sess_id, count in sorted_sessions[:10]:
                                is_query = " (QUERY SESSION)" if sess_id == session_id else ""
                                st.write(f"- `{sess_id}`: {count} matches{is_query}")

                # Combined Score Row (Top Ranked Sessions)
                st.markdown('<div class="combined-score-header">🏆 Top Sessions by Combined Score</div>', unsafe_allow_html=True)

                # Create matches list for combined score display
                # Find the best match from first region for each ranked session
                combined_matches = []
                first_region_id = region_ids[0] if region_ids else None

                for idx, ranked_item in enumerate(ranked[:20], 1):  # Show top 20
                    session_uuid = ranked_item['session_uuid']

                    # Task 1: Skip query session
                    # if session_uuid == session_id:
                    #     continue

                    # Find best match for this session from first region
                    best_match = None
                    if first_region_id and first_region_id in results:
                        for match in results[first_region_id]:
                            if not match:
                                continue

                            # Convert ScoredVector to dict if needed
                            match_dict = match_to_dict(match)
                            if not match_dict or 'id' not in match_dict:
                                continue

                            try:
                                match_id = match_dict['id']
                                if not isinstance(match_id, str):
                                    continue
                                match_session_uuid = match_id.split(".")[0]
                            except (AttributeError, KeyError, IndexError, TypeError):
                                continue

                            # # Skip query session matches
                            # if match_session_uuid == session_id:
                            #     continue

                            if match_session_uuid == session_uuid:
                                # Safely copy the match dictionary
                                best_match = dict(match_dict)
                                # Override score with combined score for display
                                best_match['score'] = ranked_item['combined_score']
                                break

                    if best_match:
                        combined_matches.append(best_match)
                    else:
                        # Create a dummy match for display if not found
                        dummy_match = {
                            'id': f"{session_uuid}.combined.0",
                            'score': ranked_item['combined_score'],
                            'metadata': ranked_item.get('metadata', {})
                        }
                        combined_matches.append(dummy_match)

                # Display combined score row
                display_image_row(combined_matches, items_per_row=5, show_rank=True)

                st.divider()

                # Individual Region Rows
                st.markdown("### 📊 Results by Region")

                tabs = st.tabs([f"Region: {region_id}" for region_id in region_ids])

                for tab, region_id in zip(tabs, region_ids):
                    with tab:
                        region_matches = results.get(region_id, [])

                        if region_matches:
                            # Task 1: Filter out query session and invalid matches
                            valid_matches = []
                            for m in region_matches:
                                if not m:
                                    continue

                                # Convert ScoredVector to dict if needed
                                match_dict = match_to_dict(m)
                                if not match_dict or 'id' not in match_dict:
                                    continue

                                # Extract session UUID from match ID
                                try:
                                    match_id = match_dict['id']
                                    if not isinstance(match_id, str):
                                        continue
                                    match_session_uuid = match_id.split(".")[0]
                                except (AttributeError, KeyError, IndexError, TypeError):
                                    continue

                                # Exclude query session
                                # if match_session_uuid == session_id:
                                #     continue

                                # Ensure score exists (Pinecone matches should have score)
                                if 'score' not in match_dict:
                                    # If no score, try to use a default or skip
                                    # But first check if it's a valid match structure
                                    if 'values' in match_dict:
                                        # This might be a vector match without score, skip it
                                        continue
                                    else:
                                        # Add default score for display
                                        match_dict['score'] = 0.0

                                valid_matches.append(match_dict)

                            if not valid_matches:
                                st.warning(f"No valid matches found for region **{region_id}** (after filtering query session)")
                                st.info(f"Original matches: {len(region_matches)}")

                                # Debug: Show first match structure if available
                                if region_matches:
                                    with st.expander("Debug: First match structure"):
                                        first_match = region_matches[0]
                                        match_dict = match_to_dict(first_match)

                                        # Safely display match structure
                                        try:
                                            debug_info = {
                                                'match_type': str(type(first_match)),
                                                'id': str(match_dict.get('id', 'N/A')) if match_dict else 'N/A',
                                                'has_score': 'score' in match_dict if match_dict else False,
                                                'score': float(match_dict.get('score', 0)) if match_dict and 'score' in match_dict else 'N/A',
                                                'has_metadata': 'metadata' in match_dict if match_dict else False,
                                                'metadata_keys': list(match_dict.get('metadata', {}).keys()) if match_dict and isinstance(match_dict.get('metadata'), dict) else 'N/A',
                                                'all_keys': [str(k) for k in match_dict.keys()] if match_dict else 'N/A'
                                            }
                                            st.json(debug_info)

                                            # Also show raw match ID extraction
                                            if match_dict:
                                                match_id = match_dict.get('id')
                                                if match_id:
                                                    extracted_session = match_id.split('.')[0] if isinstance(match_id, str) else 'N/A'
                                                    st.code(f"Match ID: {match_id}\nExtracted Session: {extracted_session}\nQuery Session: {session_id}\nIs Query Session: {extracted_session == session_id}")
                                        except Exception as e:
                                            st.error(f"Error displaying debug info: {e}")
                                            st.text(f"Match type: {type(first_match)}")
                                            if match_dict:
                                                st.text(f"Match dict keys: {list(match_dict.keys())}")
                            else:
                                # st.info(f"Found {len(valid_matches)} matches for region **{region_id}** (excluding query session)")

                                # Sort matches by score (descending)
                                valid_matches = sorted(valid_matches, key=lambda x: x.get('score', 0.0), reverse=True)

                                # Display images in rows
                                display_image_row(valid_matches, items_per_row=5, show_rank=True)

                                # Optional: Show data in expandable section
                                with st.expander("View Raw Data"):
                                    df_data = []
                                    for idx, match in enumerate(valid_matches[:20], 1):  # Show first 20 in table
                                        if not match:
                                            continue
                                        match_dict = match_to_dict(match) if not isinstance(match, dict) else match
                                        if match_dict:
                                            metadata = match_dict.get('metadata', {})
                                            match_id = match_dict.get('id', 'unknown')
                                            match_score = match_dict.get('score', 0.0)
                                            df_data.append({
                                                'Rank': idx,
                                                'Session ID': match_id.split(".")[0] if isinstance(match_id, str) and '.' in match_id else str(match_id),
                                                'Score': f"{match_score:.6f}",
                                                'S3 Bucket': metadata.get('s3_bucket', 'N/A') if isinstance(metadata, dict) else 'N/A',
                                                'S3 Key': metadata.get('s3_key', 'N/A')[:50] + '...' if isinstance(metadata, dict) and metadata.get('s3_key') else 'N/A',
                                                'Full ID': str(match_id)
                                            })
                                    if df_data:
                                        df = pd.DataFrame(df_data)
                                        st.dataframe(df, use_container_width=True)
                                    else:
                                        st.info("No data to display")
                        else:
                            st.warning(f"No matches found for region {region_id}")

                # Summary Statistics
                with st.expander("📈 Summary Statistics"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Top 5 Sessions by Combined Score:**")
                        for idx, item in enumerate(ranked[:5], 1):
                            session_url = f"https://dashboard.app.entrupy-internal.com/detail/{item['session_uuid'][:]}"
                            st.markdown(f"{idx}. Session: [{item['session_uuid']}]({session_url}) Combined Score: {item['combined_score']:.4f} Average Score: {item['average_score']:.4f}", unsafe_allow_html=True)
                            st.write("---")

                    with col2:
                        st.markdown("**Common Sessions Across Regions:**")
                        common_sessions = fetch_common_session_id(results)
                        st.write(f"Found {len(common_sessions)} unique sessions")
                        for session in common_sessions[:10]:
                            st.write(f"• `{session[:]}`")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    else:
        # Initial state - show instructions
        st.info("👈 Enter search parameters in the sidebar and click 'Search' to begin")

        # Show sample usage
        with st.expander("📖 How to Use"):
            st.markdown("""
            1. **Session ID**: Enter the UUID of the session you want to search
            2. **Brand ID**: Specify the brand (e.g., 'louis_vuitton', 'gucci', etc.)
            3. **Region IDs**: Enter the regions to search, one per line
            4. **Top K**: Select how many results to retrieve per region
            5. Click **Search** to see results

            **Features:**
            - 🏆 Combined score row shows top-ranked sessions across all regions
            - 📊 Individual region tabs show results for each specific region
            - 🔄 Horizontal scrolling for easy navigation through results
            - 📈 Summary statistics for quick insights
            """)

if __name__ == "__main__":
    main()