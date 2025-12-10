function z = smooth_abs_max_directed(X, listUnique)
%% 有向超图的Max-Absolute变分 (对应论文中的Ω3)
% 根据公式(13)计算：
% z_ei = max { ||[X_k,: - X_o,:]_+||_1, ||X_o,: - X_j,:||_1, ||X_k,: - X_l,:||_1 }
% 其中：
%   - 第一项：tail → head的距离（带投影）
%   - 第二项：tail内部节点之间的距离
%   - 第三项：head内部节点之间的距离
%
% 输入:
%   X: N×P 信号矩阵，每行是一个节点的时间序列信号
%   listUnique: D×1 cell数组，每个元素是一个struct
%               struct.head: head节点索引（被指向的节点，I_e）
%               struct.tail: tail节点索引（指出的节点，O_e）
% 输出:
%   z: D×1 距离向量，每个元素是对应超边的最大绝对距离

D = length(listUnique);  % 候选超边的数量
z = zeros(D, 1);         % 初始化距离向量

% 对每个候选超边计算距离
for i = 1:D
    edge = listUnique{i};  % 获取第i个超边
    
    % 提取head和tail节点
    if isstruct(edge)
        I_e = edge.head;   % Head nodes (in-nodes, 被指向的节点)
        O_e = edge.tail;   % Tail nodes (out-nodes, 指出的节点)
    elseif iscell(edge) && length(edge) == 2
        I_e = edge{1};     % 假设第一个元素是head
        O_e = edge{2};     % 假设第二个元素是tail
    else
        error('listUnique的格式不正确！每个元素应该是struct(head, tail)或{head, tail}');
    end
    
    % 确保是行向量
    if iscolumn(I_e)
        I_e = I_e';
    end
    if iscolumn(O_e)
        O_e = O_e';
    end
    
    temp = [];  % 存储所有距离，用于后续比较取最大值
    
    %% 第一部分：Tail → Head 的距离（带投影 [·]_+）
    % 对应公式中的 ||[X_k,: - X_o,:]_+||_1
    for o = O_e  % 遍历所有tail节点
        for k = I_e  % 遍历所有head节点
            % 计算差值
            diff = X(k,:) - X(o,:);
            
            % 应用投影 [·]_+ = max(0, ·)
            diff_projected = max(0, diff);
            
            % 计算L1范数（绝对值之和）
            distance = sum(abs(diff_projected));
            
            % 存储这个距离
            temp(end+1) = distance;
        end
    end
    
    %% 第二部分：Tail内部节点之间的距离
    % 对应公式中的 ||X_o,: - X_j,:||_1，其中 v_o, v_j ∈ O_e
    if length(O_e) > 1  % 只有当tail有多个节点时才计算
        iter = 2;  % 用于处理节点组合
        for a = 1:length(O_e)-1  % 从第1个tail节点到倒数第2个
            for b = iter:length(O_e)  % 从第2个到最后一个，避免重复
                % 获取节点索引
                node_a = O_e(a);
                node_b = O_e(b);
                
                % 计算L1范数距离
                distance = sum(abs(X(node_a,:) - X(node_b,:)));
                
                % 存储这个距离
                temp(end+1) = distance;
            end
            iter = iter + 1;
        end
    end
    
    %% 第三部分：Head内部节点之间的距离
    % 对应公式中的 ||X_k,: - X_l,:||_1，其中 v_k, v_l ∈ I_e
    if length(I_e) > 1  % 只有当head有多个节点时才计算
        iter = 2;  % 用于处理节点组合
        for a = 1:length(I_e)-1  % 从第1个head节点到倒数第2个
            for b = iter:length(I_e)  % 从第2个到最后一个，避免重复
                % 获取节点索引
                node_a = I_e(a);
                node_b = I_e(b);
                
                % 计算L1范数距离
                distance = sum(abs(X(node_a,:) - X(node_b,:)));
                
                % 存储这个距离
                temp(end+1) = distance;
            end
            iter = iter + 1;
        end
    end
    
    %% 取所有距离的最大值
    % 对应公式(13)中的 max 操作
    if isempty(temp)
        z(i) = 0;  % 如果没有计算任何距离，设为0
    else
        z(i) = max(temp);  % 存储最大绝对距离
    end
end

end