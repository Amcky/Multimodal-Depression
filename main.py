import glob  # glob库，用于查找符合特定规则的文件路径名
from tqdm import trange, tqdm  # tqdm库，用于显示进度条 (trange 是 tqdm(range(...)) 的简写)
import torch  # PyTorch深度学习框架
import shutil  # shutil库，用于文件操作，如复制、移动、删除 (在此脚本中并未直接使用)
from pytorch_lightning import Trainer, seed_everything  # PyTorch Lightning库，用于简化PyTorch训练流程
from dataset import VideoDataset, VideoRegressionDataModule  # 从自定义的dataset.py文件中导入数据集和数据模块类
from model import VideoRegressionModel  # 从自定义的model.py文件中导入模型类
import os  # os库，用于与操作系统交互，如路径操作
import argparse  # argparse库，用于解析命令行参数
import pytorch_lightning as pl  # PyTorch Lightning的别名
from lightning.pytorch.loggers import CSVLogger  # PyTorch Lightning的CSV日志记录器
from lightning.pytorch.tuner import Tuner  # PyTorch Lightning的Tuner模块，用于自动调整超参数 (在此脚本中并未直接使用Tuner的find_optimal_lr等功能)
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping  # PyTorch Lightning的回调函数：学习率监视器和早停法
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint  # PyTorch Lightning的回调函数：模型检查点保存

seed = 1  # 设置随机种子以保证实验的可复现性
import re  # re库，用于正则表达式操作

seed_everything(seed)  # PyTorch Lightning提供的函数，用于设置Python、NumPy和PyTorch的随机种子


def find_min_mae_file(file_list):
    """
    从给定的文件路径列表中找到文件名中包含最小val_mae_epoch值的文件。

    Args:
        file_list (list): 包含文件路径字符串的列表。
                          文件名应符合包含 "val_mae_epoch=数值" 的模式。

    Returns:
        str or None: 具有最小val_mae_epoch值的文件路径。如果未找到匹配文件或列表为空，则返回None。
    """
    min_mae = float('inf')  # 初始化最小MAE为一个极大值
    min_mae_file = None  # 初始化具有最小MAE的文件路径为None

    # 定义正则表达式模式，用于从文件名中提取 val_mae_epoch 的值
    # r'val_mae_epoch=([\d.]+)' 匹配 "val_mae_epoch=" 后跟一个或多个数字或小数点（捕获这部分数值）
    mae_pattern = re.compile(r'val_mae_epoch=([\d.]+)')

    for file_path in file_list:  # 遍历文件路径列表
        match = mae_pattern.search(file_path)  # 在当前文件路径中搜索模式
        if match:  # 如果找到匹配
            mae_value = float(match.group(1))  # 提取捕获到的MAE值并转换为浮点数
            if mae_value < min_mae:  # 如果当前MAE值小于已记录的最小MAE值
                min_mae = mae_value  # 更新最小MAE值
                min_mae_file = file_path  # 更新具有最小MAE的文件路径

    return min_mae_file  # 返回找到的最佳文件路径


if __name__ == '__main__':
    # # def Encoder(): # 这行被注释掉了，似乎是一个未完成或已移除的函数定义
    parser = argparse.ArgumentParser(description='视频回归模型训练脚本')  # 创建命令行参数解析器

    # 定义命令行参数
    # --- 数据相关参数 ---
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!例如你的数据在/Users/wangyaowei/PycharmProjects/PreTrain/课程大作业_抑郁症/AVEC2014
    #你就把/Users/wangyaowei/PycharmProjects/PreTrain/课程大作业_抑郁症写在data_dir
    parser.add_argument('--data_dir', default='/wywconda/Pycharm/课程大作业_抑郁症', type=str,
                        help='预处理后的视频帧数据所在的根目录')
    parser.add_argument('--label_file', default='dataset.csv', type=str,
                        help='包含标签信息的CSV文件路径')
    parser.add_argument('--train_data', default=['AVEC2014-train'], nargs='+',
                        help='用于训练的数据集名称列表 (对应data_dir下的子目录名)')
    parser.add_argument('--val_data', default=['AVEC2014-test'], nargs='+', help='用于验证的数据集名称列表')
    parser.add_argument('--test_data', default=['AVEC2014-test'], nargs='+', help='用于测试的数据集名称列表')

    # --- 模型/数据处理参数 ---
    parser.add_argument('--num_frames', default=1, type=int,
                        help='每个视频样本输入到模型的帧数 (如果模型处理序列，则为序列长度)')
    parser.add_argument('--frame_interval', default=1, type=int, help='采样视频帧时的间隔')

    ####要与数据预处理中的type保持一致
    parser.add_argument('--type', default='dlib', type=str,
                        help='数据预处理类型或特征类型 (用于构成数据子目录名)')
    parser.add_argument('--pretrain', default='webface', type=str,
                        help='预训练模型的类型或权重来源 (例如 "webface", "imagenet")')
    parser.add_argument('--save_dir', default='iresnet50', type=str,
                        help='模型检查点和日志的保存目录 (通常是模型名称或实验标识)')
    parser.add_argument('--remove_rate', default=0.1, type=float,
                        help='一个自定义参数，可能用于数据增强或特征选择中的移除比例 (具体用途需看VideoRegressionModel)')
    parser.add_argument('--top_rate', default=0.5, type=float,
                        help='一个自定义参数，可能用于选择top-k特征或样本 (具体用途需看VideoRegressionModel)')

    # --- 训练过程参数 ---
    parser.add_argument('--batch_size', default=128, type=int, help='训练和评估时的批量大小')
    parser.add_argument('--num_workers', default=8, type=int, help='数据加载器使用的工作进程数')
    parser.add_argument('--max_epochs', default=20, type=int, help='最大训练轮数')
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='优化器的初始学习率')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='优化器的权重衰减系数 (L2正则化)')
    parser.add_argument('--dropout_rate', default=0.7, type=float, help='模型中Dropout层的丢弃率')
    parser.add_argument("--deviceid", nargs="+", default=[0], type=int, help="使用的GPU设备ID列表 (例如 [0], [0, 1])")

    args = parser.parse_args()  # 解析命令行参数

    # --- 配置PyTorch Lightning回调函数 ---
    # 模型检查点回调：用于在训练过程中保存最佳模型
    # monitor="val_mae_epoch": 监控验证集上的MAE指标 (epoch级别)
    # mode='min': MAE越小越好，所以模式是'min'
    # filename='encoder_{epoch}-{val_mae_epoch:.2f}-{val_rmse_epoch:.2f}': 保存的文件名格式
    checkpoint_callback = ModelCheckpoint(
        monitor="val_mae_epoch",
        mode='min',
        filename='encoder_{epoch}-{val_mae_epoch:.2f}-{val_rmse_epoch:.2f}'
    )

    # 早停回调：用于在验证集性能不再提升时提前停止训练，防止过拟合
    # monitor='val_loss_epoch': 监控验证集上的损失 (epoch级别)
    # mode='min': 损失越小越好
    # patience=10: 如果连续10个epoch验证损失没有改善，则停止训练
    early_stopping_callback = EarlyStopping(
        monitor='val_loss_epoch',
        mode='min',
        patience=10
    )

    print(args)  # 打印解析后的参数配置

    # --- 初始化模型和数据模块 ---
    model = VideoRegressionModel(args)  # 创建视频回归模型实例，传入解析的参数
    datamodel = VideoRegressionDataModule(args)  # 创建数据模块实例，传入解析的参数

    # --- 初始化PyTorch Lightning Trainer ---
    trainer = Trainer(
        max_epochs=args.max_epochs,  # 最大训练轮数
        accelerator="cpu",  # 使用GPU进行加速
        # strategy='ddp',  # 使用DistributedDataParallel策略进行多GPU训练 (如果args.deviceid多于1个)
        # 如果只有一个GPU，DDP会自动退化为单GPU训练或给出警告。
        # 如果args.deviceid只有一个元素，'ddp'可能不是最佳选择，可以直接用默认或不指定。
        check_val_every_n_epoch=True,  # 每个epoch结束后都进行验证 (原文是True，应为整数，如1)
        # 改为1，表示每个epoch都验证
        # limit_train_batches=0.1, # (可选，被注释) 限制每个epoch训练的批次数，用于快速调试
        # limit_val_batches=0.1,   # (可选，被注释) 限制每个epoch验证的批次数，用于快速调试
        # devices=args.deviceid,  # 指定使用的GPU设备
        logger=CSVLogger(save_dir=args.train_data[0]),  # 使用CSVLogger记录训练指标，日志保存在以第一个训练数据集命名的目录下
        # 注意: save_dir最好是一个更通用的路径，如args.save_dir，而不是依赖于args.train_data[0]
        callbacks=[checkpoint_callback, early_stopping_callback]  # 添加模型检查点和早停回调
    )

    # --- 开始训练 ---
    trainer.fit(model, datamodel)  # 使用指定模型和数据模块开始训练过程

    # --- 训练结束后，加载最佳模型并进行测试 ---
    # trainer.loggers[0].root_dir 获取CSVLogger的根保存目录
    # glob.glob(...) 查找该目录下所有子目录中（通常是version_X/checkpoints/）的.ckpt检查点文件
    checkpoints = glob.glob(os.path.join(trainer.loggers[0].root_dir, '*', '*', '*.ckpt'))
    if not checkpoints:  # 如果找不到ckpt文件，尝试上一级目录
        checkpoints = glob.glob(os.path.join(trainer.loggers[0].root_dir, '*', '*.ckpt'))

    best_model_path = find_min_mae_file(checkpoints)  # 从找到的检查点文件中选出MAE最小的那个
    print(f"找到的最佳模型路径: {best_model_path}")

    if best_model_path:
        # 从最佳检查点文件加载模型权重
        # args参数需要传递给模型类的构造函数，因为它可能依赖于这些参数来构建模型结构
        best_model = VideoRegressionModel.load_from_checkpoint(best_model_path, args=args)

        # 另一种获取最佳模型路径的方式是直接使用ModelCheckpoint回调的属性 (如果训练正常完成)
        # best_model_from_callback_path = checkpoint_callback.best_model_path
        # print(f"通过回调获取的最佳模型路径: {best_model_from_callback_path}")
        # best_model = VideoRegressionModel.load_from_checkpoint(best_model_from_callback_path, args=args)

        # 使用加载的最佳模型（best_model）或训练结束时的模型（model）进行测试
        # 通常我们希望用最佳模型进行测试
        print("使用加载的最佳模型进行测试...")
        trainer.test(best_model, datamodule=datamodel)
    else:
        print("未找到任何检查点文件，无法加载最佳模型进行测试。")
        print("使用训练结束时的模型进行测试...")
        trainer.test(model, datamodule=datamodel)  # 如果找不到最佳模型，则用训练结束时的模型测试