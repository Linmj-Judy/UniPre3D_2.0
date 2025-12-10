# group layer: find neighbors for each point
# knn, knn_sparse, ball query

# gather layer, gather features by index
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function
from openpoints.cpp import pointnet2_cuda

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.zeros(B, C, nfeatures, nsample, dtype=torch.float32, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.zeros(B, C, npoint, dtype=torch.float32, device=features.device)

        pointnet2_cuda.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        # Ensure tensors are float32, contiguous, and on the same device
        device = xyz.device
        xyz = xyz.contiguous().to(device).float()
        new_xyz = new_xyz.contiguous().to(device).float()
        
        # Verify tensor properties before CUDA call
        assert new_xyz.is_contiguous(), f"new_xyz is not contiguous: {new_xyz.is_contiguous()}"
        assert xyz.is_contiguous(), f"xyz is not contiguous: {xyz.is_contiguous()}"
        assert xyz.dtype == torch.float32, f"xyz dtype is {xyz.dtype}, expected float32"
        assert new_xyz.dtype == torch.float32, f"new_xyz dtype is {new_xyz.dtype}, expected float32"
        assert xyz.device == new_xyz.device, f"Device mismatch: xyz on {xyz.device}, new_xyz on {new_xyz.device}"

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.zeros(B, npoint, nsample, dtype=torch.int32, device=device)
        
        # CUDA synchronization before CUDA extension call
        if xyz.is_cuda:
            torch.cuda.synchronize()
        
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        
        # CUDA synchronization after CUDA extension call
        if xyz.is_cuda:
            torch.cuda.synchronize()
        
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # Ensure inputs are float32, contiguous and on the same device
        device = query_xyz.device
        # Convert to float32 explicitly - CUDA extensions require float32
        query_xyz = query_xyz.contiguous().to(device).float()
        support_xyz = support_xyz.contiguous().to(device).float()
        if features is not None:
            features = features.contiguous().to(device)
        
        # Verify tensor properties
        assert query_xyz.dtype == torch.float32, f"query_xyz dtype is {query_xyz.dtype}, expected float32"
        assert support_xyz.dtype == torch.float32, f"support_xyz dtype is {support_xyz.dtype}, expected float32"
        assert query_xyz.is_contiguous(), f"query_xyz is not contiguous"
        assert support_xyz.is_contiguous(), f"support_xyz is not contiguous"
        
        # CUDA synchronization before ball_query
        if torch.cuda.is_available() and query_xyz.is_cuda:
            torch.cuda.synchronize()
            print(f"[QueryAndGroup] 准备调用 ball_query...")
            print(f"[QueryAndGroup] query_xyz shape: {query_xyz.shape}, dtype: {query_xyz.dtype}, device: {query_xyz.device}, is_contiguous: {query_xyz.is_contiguous()}")
            print(f"[QueryAndGroup] support_xyz shape: {support_xyz.shape}, dtype: {support_xyz.dtype}, device: {support_xyz.device}, is_contiguous: {support_xyz.is_contiguous()}")
            print(f"[QueryAndGroup] radius: {self.radius}, nsample: {self.nsample}")
        
        try:
            print(f"[QueryAndGroup] 调用 ball_query(radius={self.radius}, nsample={self.nsample}, xyz=support_xyz, new_xyz=query_xyz)...")
            # ball_query expects (radius, nsample, xyz, new_xyz)
            # where xyz is (B, N, 3) and new_xyz is (B, npoint, 3)
            idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)
            print(f"[QueryAndGroup] ball_query 返回成功")
        except Exception as e:
            import traceback
            print(f"[QueryAndGroup] ball_query 失败: {e}")
            print(f"[QueryAndGroup] traceback:\n{traceback.format_exc()}")
            if torch.cuda.is_available() and query_xyz.is_cuda:
                print(f"[QueryAndGroup] CUDA error: {torch.cuda.get_last_error()}")
            raise
        
        # Ensure idx is contiguous (critical for CUDA operations)
        if not idx.is_contiguous():
            idx = idx.contiguous()
        print(f"[QueryAndGroup] idx shape: {idx.shape}, device: {idx.device}, dtype: {idx.dtype}, is_contiguous: {idx.is_contiguous()}")

        if self.return_only_idx:
            return idx
        
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        print(f"[QueryAndGroup] xyz_trans shape: {xyz_trans.shape}, device: {xyz_trans.device}, is_contiguous: {xyz_trans.is_contiguous()}")
        
        # CUDA synchronization before grouping_operation
        if torch.cuda.is_available() and xyz_trans.is_cuda:
            torch.cuda.synchronize()
        
        try:
            print(f"[QueryAndGroup] 调用 grouping_operation(xyz_trans, idx)...")
            grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
            print(f"[QueryAndGroup] grouping_operation 返回成功")
            print(f"[QueryAndGroup] grouped_xyz shape: {grouped_xyz.shape}, device: {grouped_xyz.device}, is_contiguous: {grouped_xyz.is_contiguous()}")
        except Exception as e:
            import traceback
            print(f"[QueryAndGroup] grouping_operation 失败: {e}")
            print(f"[QueryAndGroup] traceback:\n{traceback.format_exc()}")
            if torch.cuda.is_available() and xyz_trans.is_cuda:
                print(f"[QueryAndGroup] CUDA error: {torch.cuda.get_last_error()}")
            raise
        
        if self.relative_xyz:
            query_xyz_trans = query_xyz.transpose(1, 2).contiguous()
            grouped_xyz = grouped_xyz - query_xyz_trans.unsqueeze(-1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius
        
        grouped_features = None
        if features is not None:
            # Ensure features is contiguous before grouping
            if not features.is_contiguous():
                features = features.contiguous()
            print(f"[QueryAndGroup] features shape: {features.shape}, device: {features.device}, is_contiguous: {features.is_contiguous()}")
            try:
                print(f"[QueryAndGroup] 调用 grouping_operation(features, idx)...")
                grouped_features = grouping_operation(features, idx)
                print(f"[QueryAndGroup] grouping_operation(features) 返回成功")
                print(f"[QueryAndGroup] grouped_features shape: {grouped_features.shape}, device: {grouped_features.device}, is_contiguous: {grouped_features.is_contiguous()}")
            except Exception as e:
                import traceback
                print(f"[QueryAndGroup] grouping_operation(features) 失败: {e}")
                print(f"[QueryAndGroup] traceback:\n{traceback.format_exc()}")
                if torch.cuda.is_available() and features.is_cuda:
                    print(f"[QueryAndGroup] CUDA error: {torch.cuda.get_last_error()}")
                raise
        
        return grouped_xyz, grouped_features


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        # Ensure inputs are contiguous and on the same device
        device = query_xyz.device
        query_xyz = query_xyz.contiguous().to(device)
        support_xyz = support_xyz.contiguous().to(device)
        if features is not None:
            features = features.contiguous().to(device)
        
        # CUDA synchronization before KNN
        if torch.cuda.is_available() and query_xyz.is_cuda:
            torch.cuda.synchronize()
        
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        
        # Ensure idx is contiguous (critical for CUDA operations)
        if not idx.is_contiguous():
            idx = idx.contiguous()
        
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        
        # CUDA synchronization before grouping_operation
        if torch.cuda.is_available() and xyz_trans.is_cuda:
            torch.cuda.synchronize()
        
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            query_xyz_trans = query_xyz.transpose(1, 2).contiguous()
            grouped_xyz -= query_xyz_trans.unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            # Ensure features is contiguous before grouping
            if not features.is_contiguous():
                features = features.contiguous()
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper


if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    # --------------- debug downsampling
    from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample

    npoints = 10000
    # rs = RandomSample(num_to_sample=npoints)
    # query, _= rs(points)
    idx = random_sample(points, npoints)
    # torch gather is faster then operation gather. 
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    # # --------------- debug KNN
    # knn = KNN(k=K, transpose_mode=True)
    # # knn to get the neighborhood

    # # compare time usage.
    # st = time.time()
    # for _ in range(100):
    #     _, knnidx = knn(points, query) # B G M
    #     idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    #     idx = knnidx + idx_base
    #     idx = idx.view(-1)
    #     neighborhood = points.view(B * N, -1)[idx, :]
    #     neighborhood = neighborhood.view(B, npoints, K, 3).contiguous()
    #     # normalize
    #     neighborhood1 = neighborhood - query.unsqueeze(2)
    # print(time.time() - st)
    # # print(neighborhood1.shape, '\n', neighborhood1)

    # knngroup = KNNGroup(K)
    # # KNN Group is faster then above torch indexing when warpped in class.  
    # st = time.time()
    # for _ in range(100):
    #     neighborhood2 = knngroup(query, points)
    # print(time.time() - st)
    # # print(neighborhood2.shape, '\n', neighborhood2)
    # flag = torch.allclose(neighborhood1, neighborhood2.permute(0, 2, 3, 1))
    # print(flag)

    # ------------- debug ball query
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        # ball querying is 40 times faster then KNN 
        features = query_group(query, points)
    print(time.time() - st)
    print(features.shape)
