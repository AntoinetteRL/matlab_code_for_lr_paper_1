function S = make_S(listUnique, N)
% MAKE_S  Build stacked directed incidence matrix S = [Sin; Sout]
% 兼容三种 listUnique 元素格式：
%   1) 数值行向量: [tail1 tail2 ... head]      ← 我们新格式（head 为最后一个）
%   2) struct:     有字段 'head' 和 'tail'（tail 可为向量）
%   3) cell:       {head, tail} 或 {tail, head}（tail 可为标量或向量）
%
% 输出:
%   S : (2N) x Da  稀疏矩阵，Sin 在上，Sout 在下

    Da = numel(listUnique);
    if N <= 0 || Da == 0
        S = sparse(2*N, Da);
        return;
    end

    Sin  = sparse(N, Da);
    Sout = sparse(N, Da);

    for e = 1:Da
        item = listUnique{e};

        % ---- 解析 head / tails ----
        head = [];
        tails = [];

        if isnumeric(item) && isvector(item)
            ed = item(:).';
            if numel(ed) >= 2
                head  = ed(end);
                tails = ed(1:end-1);
            end

        elseif isstruct(item)
            if isfield(item,'head') && isfield(item,'tail')
                head  = item.head;
                tails = item.tail;
            end

        elseif iscell(item)
            if numel(item) == 2
                a = item{1}; b = item{2};
                if isscalar(a) && ~isempty(b)
                    head = a; tails = b;
                elseif isscalar(b) && ~isempty(a)
                    head = b; tails = a;
                end
            end
        end

        % ---- 安全过滤 ----
        if isempty(head) || isempty(tails),   continue; end
        tails = tails(:).';
        tails = tails(tails ~= head & tails >= 1 & tails <= N);
        if isempty(tails) || head < 1 || head > N,  continue; end

        % ---- 写入入/出关联 ----
        Sin(head, e) = 1;
        Sout(tails, e) = 1;
    end

    S = [Sin; Sout];
end
