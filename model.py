import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

class EnhancedLayoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )
        
    def forward(self, x):
        return self.fc_layers(x)

def calculate_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
    x_left = torch.max(x1, x2)
    y_bottom = torch.max(y1, y2)
    x_right = torch.min(x1 + w1, x2 + w2)
    y_top = torch.min(y1 + h1, y2 + h2)
    return torch.clamp(x_right - x_left, min=0) * torch.clamp(y_top - y_bottom, min=0)

def improved_loss(outputs, y_true, X_train, alpha=0.5, beta=0.3):
    mse_loss = nn.MSELoss()(outputs, y_true)
    
    room_dims = X_train[:, :2]
    furniture_dims = X_train[:, 2:8]
    
    sofa_x, sofa_y = outputs[:, 0], outputs[:, 1]
    table_x, table_y = outputs[:, 2], outputs[:, 3]
    chair_x, chair_y = outputs[:, 4], outputs[:, 5]
    
    # Boundary constraints with margin
    margin = 0.1
    room_w, room_h = room_dims[:, 0], room_dims[:, 1]
    
    boundary_violation = (
        torch.relu(sofa_x + furniture_dims[:,0] - (room_w - margin)) + 
        torch.relu(sofa_y + furniture_dims[:,1] - (room_h - margin)) +
        torch.relu(-sofa_x + margin) + 
        torch.relu(-sofa_y + margin)
    )
    
    # Enhanced overlap penalty
    overlap_st = calculate_overlap(
        sofa_x, sofa_y, furniture_dims[:,0], furniture_dims[:,1],
        table_x, table_y, furniture_dims[:,2], furniture_dims[:,3]
    )
    overlap_sc = calculate_overlap(
        sofa_x, sofa_y, furniture_dims[:,0], furniture_dims[:,1],
        chair_x, chair_y, furniture_dims[:,4], furniture_dims[:,5]
    )
    overlap_tc = calculate_overlap(
        table_x, table_y, furniture_dims[:,2], furniture_dims[:,3],
        chair_x, chair_y, furniture_dims[:,4], furniture_dims[:,5]
    )
    
    total_constraint_loss = (
        boundary_violation.mean() + 
        2.0 * (overlap_st.mean() + overlap_sc.mean() + overlap_tc.mean())
    )
    
    return alpha * mse_loss + beta * total_constraint_loss

def train_model():
    df = pd.read_csv('data/synthetic_data.csv')
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8:].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    model = EnhancedLayoutModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    best_loss = float('inf')
    epochs = 1000
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = improved_loss(outputs, y_train, X_train)
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = improved_loss(val_outputs, y_test, X_test)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/trained_model.pth')
        
        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch+1:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}')
    
if __name__ == '__main__':
    train_model()