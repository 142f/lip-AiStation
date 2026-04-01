import shutil
from pathlib import Path


def read_ids(file_path):
    """读取 ID 列表文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def copy_files_by_ids(source_dir, target_dir, id_list, file_extension='.mp4'):
    """根据 ID 列表复制文件"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    not_found_count = 0
    not_found_files = []
    
    for metadata_id in id_list:
        source_file = source_path / f"{metadata_id}{file_extension}"
        if source_file.exists():
            shutil.copy2(source_file, target_path)
            copied_count += 1
        else:
            not_found_count += 1
            not_found_files.append(f"{metadata_id}{file_extension}")
    
    return copied_count, not_found_count, not_found_files


def main():
    # 源目录
    source_real_dir = r"E:\data\AVLips v1.0\AVLips\0_real"
    source_fake_dir = r"E:\data\AVLips v1.0\AVLips\1_fake"
    
    # 目标目录
    target_dir = r"E:\data\AVlips-test"
    
    # ID 列表文件
    real_ids_file = r"E:\Project\lip-AiStation\data\test_pro_test_id_export\0_real_metadata_ids.txt"
    fake_ids_file = r"E:\Project\lip-AiStation\data\test_pro_test_id_export\1_fake_metadata_ids.txt"
    
    # 读取 ID 列表
    real_ids = read_ids(real_ids_file)
    fake_ids = read_ids(fake_ids_file)
    
    print(f"读取到 {len(real_ids)} 个 real 样本 ID")
    print(f"读取到 {len(fake_ids)} 个 fake 样本 ID")
    
    # 创建目标子目录
    target_real_dir = Path(target_dir) / "0_real"
    target_fake_dir = Path(target_dir) / "1_fake"
    
    # 复制 real 文件
    print("\n开始复制 real 文件...")
    real_copied, real_not_found, real_missing = copy_files_by_ids(
        source_real_dir, target_real_dir, real_ids
    )
    print(f"Real 文件复制完成: 成功 {real_copied} 个, 未找到 {real_not_found} 个")
    
    if real_missing:
        print(f"未找到的 real 文件: {real_missing[:10]}{'...' if len(real_missing) > 10 else ''}")
    
    # 复制 fake 文件
    print("\n开始复制 fake 文件...")
    fake_copied, fake_not_found, fake_missing = copy_files_by_ids(
        source_fake_dir, target_fake_dir, fake_ids
    )
    print(f"Fake 文件复制完成: 成功 {fake_copied} 个, 未找到 {fake_not_found} 个")
    
    if fake_missing:
        print(f"未找到的 fake 文件: {fake_missing[:10]}{'...' if len(fake_missing) > 10 else ''}")
    
    # 总结
    print("\n" + "="*50)
    print(f"总计复制完成:")
    print(f"  Real: {real_copied}/{len(real_ids)}")
    print(f"  Fake: {fake_copied}/{len(fake_ids)}")
    print(f"  总计: {real_copied + fake_copied}/{len(real_ids) + len(fake_ids)}")
    print(f"目标目录: {target_dir}")
    print("="*50)


if __name__ == "__main__":
    main()