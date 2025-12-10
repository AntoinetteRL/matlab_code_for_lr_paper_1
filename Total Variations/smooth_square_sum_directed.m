function z = smooth_square_sum_directed(X, listUnique)
%% 有向超图的Sum-Square变分 (对应论文中的Ω1)
% 根据公式(9)计算：
% z_ei = sum{||[X_k,: - X_o,:]_+||_2^2} + sum{||X_o,: - X_j,:||_2^2} + sum{||X_k,: - X_l,:||_2^2}
% 其中：
%   - 第一项：tail → head的平方距离求和（带投影）
%   - 第二项：tail内部节点之间的平方距离求和
%   - 第三项：head内部节点之间的平方距离求和
%
% 输入:
%   X: N×P 信号矩阵，每行是一个节点的时间序列信号
%   listUnique: D×1 cell数组，每个元素是一个struct
%               struct.head: head节点索引（被指向的节点，I_e）
%               struct.tail: tail节点索引（指出的节点，O_e）
% 输出:
%   z: D×1 距离向量，每个元素是对应超边的平方距离总和

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
    
    summation = 0;  % 累积所有平方距离的总和
    
    %% 第一部分：Tail → Head 的平方距离求和（带投影 [·]_+）
    % 对应公式(9)中的第一项：sum_{v_o∈O_e, v_k∈I_e} ||[X_k,: - X_o,:]_+||_2^2
    for o = O_e  % 遍历所有tail节点
        for k = I_e  % 遍历所有head节点
            % 计算差值
            diff = X(k,:) - X(o,:);
            
            % 应用投影 [·]_+ = max(0, ·)
            diff_projected = max(0, diff);
            
            % 计算L2平方范数（差值平方和）
            % ||diff_projected||_2^2 = sum(diff_projected.^2) = diff_projected * diff_projected'
            squared_distance = diff_projected * diff_projected';
            summation = summation + squared_distance;
        end
    end
    
    %% 第二部分：Tail内部节点之间的平方距离求和
    % 对应公式(9)中的第二项：sum_{v_o, v_j ∈ O_e} ||X_o,: - X_j,:||_2^2
    if length(O_e) > 1  % 只有当tail有多个节点时才计算
        iter = 2;  % 用于处理节点组合，避免重复计算
        for a = 1:length(O_e)-1  % 从第1个tail节点到倒数第2个
            for b = iter:length(O_e)  % 从第2个到最后一个
                % 获取节点索引
                node_a = O_e(a);
                node_b = O_e(b);
                
                % 计算差值
                diff = X(node_a,:) - X(node_b,:);
                
                % 计算L2平方范数并累加
                squared_distance = diff * diff';
                summation = summation + squared_distance;
            end
            iter = iter + 1;
        end
    end
    
    %% 第三部分：Head内部节点之间的平方距离求和
    % 对应公式(9)中的第三项：sum_{v_k, v_l ∈ I_e} ||X_k,: - X_l,:||_2^2
    if length(I_e) > 1  % 只有当head有多个节点时才计算
        iter = 2;  % 用于处理节点组合，避免重复计算
        for a = 1:length(I_e)-1  % 从第1个head节点到倒数第2个
            for b = iter:length(I_e)  % 从第2个到最后一个
                % 获取节点索引
                node_a = I_e(a);
                node_b = I_e(b);
                
                % 计算差值
                diff = X(node_a,:) - X(node_b,:);
                
                % 计算L2平方范数并累加
                squared_distance = diff * diff';
                summation = summation + squared_distance;
            end
            iter = iter + 1;
        end
    end
    
    %% 存储该超边的总平方距离
    z(i) = summation;
end

end