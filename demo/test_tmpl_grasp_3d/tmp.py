def read_symmetric_info(self, json_file_path: str, scale: float, rotation_step: float = 1) -> np.ndarray:
    """
    读取 CAD 模型的对称性信息      
    Args:
        json_file_path (str): json 文件路径
        scale (float): 缩放因子( 将位移单位转换到米制单位的比例 )
        rotation_step (float): 旋转步长( 仅适用于连续旋转 ), 单位: 度
    Returns:
        (np.ndarray): 对称变换矩阵 N*4*4, 如果失败则返回 None
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"failed to read symmetric info from: '{json_file_path}' \n{e}")
        return None  # 返回空数组表示读取失败
    # end with

    # 读取旋转轴( 仅适用于连续旋转 )
    continuous_tf_list = []
    if "symmetries_continuous" in data:
        symmetries = data.get('symmetries_continuous', [])

        # 遍历每个对称性定义
        for i, symmetry in enumerate(symmetries):
            axis = symmetry.get('axis', [])
            offset = symmetry.get('offset', [])

            # 将轴和偏移量转换为 numpy 数组
            axis = np.array(axis)
            offset = np.array(offset) * scale
            logging.info(f"symmetries_continuous index: {i}, axis: {axis}, offset: {offset}")

            # 生成变换矩阵
            offset = offset.reshape((3, 1))
            for angle in np.arange(0, 360, rotation_step):
                R = transforms3d.axangles.axangle2mat(axis, np.radians(angle))
                t = offset - R.dot(offset)

                # 组合旋转矩阵和平移向量
                tf = np.eye(4)
                tf[:3, :3] = R
                tf[:3, 3] = t.flatten()

                continuous_tf_list.append(tf)
            # end for
        # end for
    # end if

    # 读取离散对称性
    discrete_tf_list = []
    if "symmetries_discrete" in data:
        symmetries = data.get('symmetries_discrete', [])
        for i, symmetry in enumerate(symmetries):
            tf = np.array(symmetry).reshape((4, 4))
            tf[:3, 3] *= scale  # 缩放平移向量
            discrete_tf_list.append(tf)
        # end for
    # end if
    discrete_tf_list.append(np.eye(4))  # 添加单位矩阵作为默认变换

    # 组合离散和连续变换
    tf_list = []
    for disc_tf in discrete_tf_list:
        if len(continuous_tf_list) > 0:
            for cont_tf in continuous_tf_list:
                tf_list.append(disc_tf @ cont_tf)
            # end for
        else:
            tf_list.append(disc_tf)
        # end if
    # end for

    logging.info(f"symmetric transforms num: \033[92m{len(tf_list)}\033[0m")

    # 将变换矩阵列表转换为 numpy 数组
    tf_array = np.array(tf_list)

    return tf_array
# end def read_symmetric_info
