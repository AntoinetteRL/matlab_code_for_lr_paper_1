function z = smooth_square_max_directed(X, listUnique)
% z(e) = max_{t in tails, tau} ( x_t(tau) - x_head(tau) )^2
% 兼容三种 listUnique 元素格式：
%   1) 数值行向量: [tail1 tail2 ... head]      ← 我们新格式（head 是最后一个）
%   2) struct:     with fields 'head' and 'tail'
%   3) cell:       {head, tail} 或 {tail, head}（tail 可为标量或向量）
%
% X: N x T
% listUnique: 1 x Da cell

    [N, ~] = size(X); %#ok<ASGLU>
    Da = numel(listUnique);
    z = zeros(Da,1);

    for e = 1:Da
        item = listUnique{e};

        % ---- 解析 head / tails ----
        head = [];
        tails = [];

        if isnumeric(item) && isvector(item)
            ed = item(:).';                 % [.. head]
            if numel(ed) >= 2
                head  = ed(end);
                tails = ed(1:end-1);
            end

        elseif isstruct(item)
            % 期望字段名：head, tail（tail 可向量）
            if isfield(item,'head') && isfield(item,'tail')
                head  = item.head;
                tails = item.tail;
            end

        elseif iscell(item)
            % 尝试 {head, tail}；若第一格不是标量而第二格是，则认为 {tail, head}
            if numel(item) == 2
                a = item{1}; b = item{2};
                if isscalar(a) && ~isempty(b)
                    head = a; tails = b;
                elseif isscalar(b) && ~isempty(a)
                    head = b; tails = a;
                end
            end
        end

        % ---- 安全过滤 & 计算 ----
        if isempty(head) || isempty(tails)
            z(e) = 0; continue;
        end
        tails = tails(:).';
        tails = tails(tails ~= head & tails >= 1 & tails <= N);
        if isempty(tails) || head < 1 || head > N
            z(e) = 0; continue;
        end

        diffs2 = (X(tails,:) - X(head,:)).^2;   % (#tails) x T
        z(e) = max(diffs2, [], 'all');
    end
end
