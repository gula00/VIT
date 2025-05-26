import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomAffine, ColorJitter, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm
import os
import shutil
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from models import ClassificationVisionTransformer

class BreastCancerDataset(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        '''
        参数说明：
        root (str): 数据集所在的根目录路径
        is_train (bool): 标识是训练集还是验证集，默认为True（训练集）
        transform (callable, optional): 应用于图像的变换操作（数据增强/预处理），默认为None
        '''
        # 初始化类别名称（良性、恶性、正常）
        self.categories = ["benign", "malignant", "normal"]
        self.transform = transform  # 存储图像变换方法
        
        # 初始化存储路径和标签的列表
        self.image_paths = []  # 存储所有符合要求的图像路径
        self.labels = []       # 存储对应的类别标签（0,1,2）

        # 遍历每个类别目录收集数据
        for i, category in enumerate(self.categories):
            # 构建类别完整路径（例如：root/benign）
            category_path = os.path.join(root, category)
            
            # 使用glob查找所有带_mask后缀的PNG文件
            # 注意：实际使用需确保文件名匹配，也可修改为其他格式（如*jpg）
            for img_path in glob.glob(os.path.join(category_path, "*_mask.png")):
                self.image_paths.append(img_path)  # 记录图像路径
                self.labels.append(i)              # 记录类别索引（0,1,2）

        # 使用分层抽样分割数据集（保持类别分布）
        # test_size=0.2 表示验证集占20%
        # stratify=self.labels 确保训练/验证集的类别比例与原始数据一致
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            self.image_paths, 
            self.labels, 
            test_size=0.2,
            random_state=42,    # 固定随机种子保证可重复性
            stratify=self.labels
        )

        # 根据is_train参数选择数据集
        if is_train:
            self.image_paths = train_paths
            self.labels = train_labels
        else:
            self.image_paths = val_paths
            self.labels = val_labels

    def __len__(self):
        '''返回数据集样本总数'''
        return len(self.labels)

    def __getitem__(self, index):
        '''
        获取单个样本
        参数：index (int) - 数据索引
        返回：元组 (image, label)
        '''
        # 读取图像文件（自动处理不同格式）
        image = Image.open(self.image_paths[index]).convert("L")
        
        # 获取对应标签
        label = self.labels[index]
        
        # 应用图像变换（如有）
        if self.transform:
            image = self.transform(image)
            
        return image, label  # 返回图像张量和标签

def get_transforms(image_size=224):
    """创建增强的训练和验证的图像变换管道"""
    # 单通道图像的均值和标准差
    mean = [0.485]  # 单通道均值
    std = [0.229]   # 单通道标准差
    
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=15),
        RandomAffine(
            degrees=(-10, 10),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            shear=5
        ),
        ColorJitter(
            brightness=0.1,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        ToTensor(),
        Resize((image_size, image_size)),
        Normalize(mean=mean, std=std)  # 使用单通道的均值和标准差
    ])
    
    test_transform = Compose([
        ToTensor(),
        Resize((image_size, image_size)),
        Normalize(mean=mean, std=std)  # 使用单通道的均值和标准差
    ])
    
    return train_transform, test_transform

def train(model, train_dataloader, epoch, epochs, criterion, optimizer, device):
    """改进的模型训练函数"""
    progress_bar = tqdm(train_dataloader, colour="green")
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_metrics = []

    for iteration, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()
        
        # 记录每个batch的指标
        batch_metrics.append({
            'loss': loss.item(),
            'accuracy': (predicted == labels).float().mean().item()
        })
        
        progress_bar.set_description(
            f"TRAIN | Epoch: {epoch+1}/{epochs} | "
            f"Iter: {iteration+1}/{len(train_dataloader)} | "
            f"Loss: {loss.item():.4f} | "
            f"Acc: {100 * correct / total:.2f}%"
        )
    
    avg_loss = total_loss / len(train_dataloader)
    avg_acc = 100 * correct / total
    
    print(f"\nEpoch: {epoch+1}/{epochs} | "
          f"Train Loss: {avg_loss:.4f} | "
          f"Train Acc: {avg_acc:.2f}%")
    
    return avg_loss, avg_acc, batch_metrics

def valid(model, test_dataloader, criterion, device):
    """改进的模型验证函数"""
    model.eval()
    all_losses = []
    all_labels = []
    all_predictions = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Validating", colour="blue")
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            
            all_losses.append(loss.item())
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = np.mean(all_losses)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):  # 3个类别
        fpr[i], tpr[i], _ = roc_curve(
            np.array(all_labels) == i,
            np.array(all_probs)[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=["benign", "malignant", "normal"]))
    
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    print(f"Validation accuracy: {accuracy}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'all_losses': all_losses,
        'true_labels': all_labels,
        'pred_labels': all_predictions,
        'pred_probs': all_probs,
        'confusion_matrix': cm,
        'roc_curves': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        }
    }

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    # plt.plot(val_accs, label='Validation Accuracy')
    plt.plot([acc * 100 for acc in val_accs], label='Validation Accuracy')  # 将验证集准确率转换为百分比
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["benign", "malignant", "normal"],
                yticklabels=["benign", "malignant", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def visualize_predictions(model, dataloader, device, save_path, num_samples=5):
    """可视化模型预测结果"""
    model.eval()
    categories = ["benign", "malignant", "normal"]
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # 将图像移回CPU进行可视化
            images = images.cpu()
            
            plt.figure(figsize=(15, 3))
            for j in range(min(5, len(images))):
                plt.subplot(1, 5, j+1)
                img = images[j].squeeze().numpy()  # 移除通道维度
                img = (img * 0.229 + 0.485)  # 反标准化
                img = np.clip(img, 0, 1)
                
                plt.imshow(img, cmap='gray')  # 使用灰度图显示
                true_label = categories[labels[j]]
                pred_label = categories[predictions[j]]
                prob = probs[j][predictions[j]].item()
                
                title = f'True: {true_label}\nPred: {pred_label}\nProb: {prob:.2f}'
                plt.title(title, fontsize=8)
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'predictions_sample_{i+1}.png'))
            plt.close()

def plot_metrics(metrics, save_path):
    """绘制详细的评估指标"""
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(["benign", "malignant", "normal"]):
        plt.plot(
            metrics['roc_curves']['fpr'][i],
            metrics['roc_curves']['tpr'][i],
            label=f'{label} (AUC = {metrics["roc_curves"]["auc"][i]:.2f})'
        )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'roc_curves.png'))
    plt.close()
    
    # 保存评估指标到JSON文件
    metrics_dict = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1']),
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def main():
    # 配置参数
    data_path = "Dataset_BUSI_with_GT"
    batch_size = 16  # 增加批量大小
    image_size = 224
    epochs = 50  # 增加训练轮数
    lr = 1e-4  # 调整学习率
    patience = 10  # 早停耐心值
    log_path = "tensorboard"
    checkpoint_path = "trained_models"
    results_path = "results"
    
    # 创建必要的目录
    for path in [log_path, checkpoint_path, results_path]:
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.makedirs(path)
    
    # 获取数据变换
    train_transform, valid_transform = get_transforms(image_size)
    
    # 创建数据集和数据加载器
    train_dataset = BreastCancerDataset(
        root=data_path, 
        is_train=True, 
        transform=train_transform
    )
    valid_dataset = BreastCancerDataset(
        root=data_path, 
        is_train=False, 
        transform=valid_transform
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4,  # 增加工作进程数
        shuffle=True,
        drop_last=True
    )
    
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationVisionTransformer(
        img_size=224,  # 增加图像大小
        patch_size=16,
        in_chans=1,
        num_classes=3,
        embed_dim=768,
        depth=12,
        num_heads=12
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)  # 使用AdamW优化器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    best_acc = -1
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss, train_acc, _ = train(model, train_dataloader, epoch, epochs, 
                                       criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        val_metrics = valid(model, valid_dataloader, criterion, device)
        val_losses.append(val_metrics['loss'])
        val_accs.append(val_metrics['accuracy'])
        
        # 更新学习率
        scheduler.step(val_metrics['accuracy'])
        
        # 保存模型
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': val_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'best_accuracy': best_acc
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_path, "last.pt"))
        
        if val_metrics['accuracy'] > best_acc:
            torch.save(checkpoint, os.path.join(checkpoint_path, "best.pt"))
            best_acc = val_metrics['accuracy']
            print(f"New best model saved with accuracy: {best_acc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        # 早停检查
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
        
        print("-" * 60)
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs, results_path)
    
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(checkpoint_path, "best.pt")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 最终评估和可视化
    final_metrics = valid(model, valid_dataloader, criterion, device)
    plot_confusion_matrix(final_metrics['confusion_matrix'], results_path)
    plot_metrics(final_metrics, results_path)
    visualize_predictions(model, valid_dataloader, device, results_path)

if __name__ == "__main__":
    main() 