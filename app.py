import os
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"

import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import EnhancedLayoutModel

# Load model
model = EnhancedLayoutModel()
model.load_state_dict(torch.load('models/trained_model.pth'))
model.eval()

def format_dimensions(name, x, y, w, h):
    return f"{name.capitalize()}\n({x:.1f}, {y:.1f})\n{w:.1f}m × {h:.1f}m"

def main():
    st.title("🏠 Smart Furniture Arrangement Optimizer")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("Room Dimensions")
        room_w = st.slider("Width (meters)", 5.0, 10.0, 8.0, 0.1)
        room_h = st.slider("Height (meters)", 5.0, 10.0, 6.0, 0.1)
        
    with col2:
        st.header("Furniture Specifications")
        sofa_w = st.slider("Sofa Width", 1.0, 3.0, 2.0, 0.1)
        sofa_h = st.slider("Sofa Height", 1.0, 3.0, 1.0, 0.1)
        table_w = st.slider("Table Width", 1.0, 2.0, 1.5, 0.1)
        table_h = st.slider("Table Height", 1.0, 2.0, 1.5, 0.1)
        chair_w = st.slider("Chair Width", 0.5, 1.5, 1.0, 0.1)
        chair_h = st.slider("Chair Height", 0.5, 1.5, 1.0, 0.1)

    if st.button("✨ Generate Optimal Layout"):
        input_data = torch.FloatTensor([[
            room_w, room_h,
            sofa_w, sofa_h,
            table_w, table_h,
            chair_w, chair_h
        ]])
        
        with torch.no_grad():
            pred = model(input_data).numpy()[0]
        
        sofa_x, sofa_y, table_x, table_y, chair_x, chair_y = pred
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(0, room_w)
        ax.set_ylim(0, room_h)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("Width (meters)", labelpad=10)
        ax.set_ylabel("Height (meters)", labelpad=10)
        ax.set_title("Optimized Layout Visualization", pad=20)
        
        # Draw room boundaries
        ax.add_patch(plt.Rectangle(
            (0, 0), room_w, room_h,
            fill=False, edgecolor='navy', linewidth=2
        ))
        
        # Furniture colors and styles
        furniture_styles = {
            'sofa': {'color': '#2E86C1', 'alpha': 0.8},
            'table': {'color': '#28B463', 'alpha': 0.8},
            'chair': {'color': '#E74C3C', 'alpha': 0.8}
        }
        
        # Draw and annotate furniture
        for name, (x, y, w, h) in zip(
            ['sofa', 'table', 'chair'],
            [
                (sofa_x, sofa_y, sofa_w, sofa_h),
                (table_x, table_y, table_w, table_h),
                (chair_x, chair_y, chair_w, chair_h)
            ]
        ):
            ax.add_patch(plt.Rectangle(
                (x, y), w, h,
                **furniture_styles[name],
                edgecolor='black',
                linewidth=1.5
            ))
            # Position annotation
            ax.text(
                x + w/2, y + h/2,
                format_dimensions(name, x, y, w, h),
                ha='center', va='center',
                color='white', weight='bold',
                fontsize=8
            )
        
        # Add scale indicator in top-right corner
        scale_x = room_w - 1.5  # 1.5m from right edge
        scale_y = room_h - 0.5  # 0.5m from top
        ax.plot([scale_x, scale_x + 1], [scale_y, scale_y], 
                color='black', lw=2)
        ax.text(scale_x + 0.5, scale_y + 0.1, "1 meter",
                ha='center', va='bottom', fontsize=8)
        
        st.pyplot(fig)
        st.success("✅ Layout generated successfully!")
        
        # Display coordinates
        st.subheader("Coordinates")
        st.markdown(f"""
        | Furniture | X Position | Y Position | Width | Height |
        |-----------|------------|------------|-------|--------|
        | Sofa      | {sofa_x:.2f}m | {sofa_y:.2f}m | {sofa_w:.2f}m | {sofa_h:.2f}m |
        | Table     | {table_x:.2f}m | {table_y:.2f}m | {table_w:.2f}m | {table_h:.2f}m |
        | Chair     | {chair_x:.2f}m | {chair_y:.2f}m | {chair_w:.2f}m | {chair_h:.2f}m |
        """)

if __name__ == "__main__":
    main()