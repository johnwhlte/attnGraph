import torch


def gen_edge(feat_mat, min_dist):

    feat_tensor = feat_mat.T

    x_c, y_c= feat_tensor[0], feat_tensor[1]#, feat_tensor[2]
    x_c = x_c.repeat(x_c.shape[0], 1)
    y_c = y_c.repeat(y_c.shape[0], 1)
    #z_c = z_c.repeat(z_c.shape[0], 1)
    x_c2 = x_c.T
    y_c2 = y_c.T
    #z_c2 = z_c.T
    x_diff = x_c - x_c2
    y_diff = y_c - y_c2
    #z_diff = z_c - z_c2
    x_dist = torch.mul(x_diff, x_diff)
    y_dist = torch.mul(y_diff, y_diff)
    #z_dist = torch.mul(z_diff, z_diff)
    tot_dist = x_dist + y_dist#+ z_dist
    dist = torch.sqrt(tot_dist)

    init_mat = torch.where(dist<min_dist, dist, 0.)
    new_mat = torch.where(init_mat==0, init_mat, 1.)

    adj_mat = new_mat + torch.eye(new_mat.shape[0])
    adj_mat_opp = -1*(adj_mat - 1)
    #print(adj_mat_opp)
    x_diff_mat = torch.nan_to_num(torch.mul(adj_mat,x_diff**-1),posinf=0.0, neginf=0.0, nan=0.0)
    y_diff_mat = torch.nan_to_num(torch.mul(adj_mat,y_diff**-1),posinf=0.0, neginf=0.0, nan=0.0)
    ajd_mat_sparse = adj_mat_opp.to_sparse()
    x_diff_mat_sparse = x_diff_mat.to_sparse()
    #print(ajd_mat_sparse.mul(x_diff_mat_sparse).to_dense())
    adj_mat = adj_mat.long()

    degrees = torch.sum(adj_mat, 1)
    degrees = degrees.float()
    mean = torch.mean(degrees)

    print(degrees.max())
    print(degrees.min())

    edges = adj_mat.nonzero().t().contiguous()

    return edges, x_diff_mat.to_sparse(), y_diff_mat.to_sparse()


if __name__ == "__main__":

    feat_mat = torch.load('/home/sysiphus/bigData/snapshots/func4_43_0.07_snap.pt')
    edges,_,_ = gen_edge(feat_mat=feat_mat, min_dist=0.055)
    torch.save(edges, '/home/sysiphus/bigData/snapshots/true_edges.pt')
    print(edges.shape)