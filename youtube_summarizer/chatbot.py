import streamlit as st

# Set the page configuration
st.set_page_config(page_title="YouTube Buddy: Under Maintenance", page_icon="🔧")

# Sidebar content
with st.sidebar:
    st.markdown("## YouTube Buddy 🚀")
    st.write("Made by [Akash Kumar](https://www.linkedin.com/in/akashkumar2/).", unsafe_allow_html=True)
    st.markdown('💬 Do share your [feedback](https://f7f6zk74dit.typeform.com/to/kYw7Y8y7)!')

# Main content
st.subheader("YouTube Buddy: Streamline Your YouTube Experience")

# Add some spacing
st.markdown("### ")

# Display the "Under Maintenance" GIF
st.image("https://media1.tenor.com/m/XjWWvvKhJN8AAAAC/pilot-repair.gif", use_column_width=True)

# Display the maintenance message with emphasis
st.markdown("""
    <div style="text-align: center;">
        <h2 style="color: #FF6347;">⚠️ The application is currently under maintenance.</h2>
        <p style="font-size: 18px;">YouTube has made it harder to get subtitles of videos. We are currently working to resolve this asap</p>
    </div>
    """, unsafe_allow_html=True)

# Placeholder to display the reason for failure with a prominent box
# st.markdown("""
#     <div style="background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px;">
#         <strong>Reason:</strong> Please check back later.
#     </div>
#     """, unsafe_allow_html=True)

# Optionally, provide a contact or support link
# st.markdown("""
#     <div style="text-align: center; margin-top: 20px;">
#         If you need assistance, please contact <a href="mailto:infiniteakashe@egmail.com">support@example.com</a>.
#     </div>
#     """, unsafe_allow_html=True)
