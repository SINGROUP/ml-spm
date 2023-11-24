import torch


def test_conv_block():
    from mlspm.modules import Conv2dBlock, Conv3dBlock

    torch.manual_seed(0)
    x = torch.rand(10, 8, 12, 12)

    with torch.no_grad():
        a = Conv2dBlock(8, 4, kernel_size=3, depth=1, res_connection=False, activation=torch.nn.ReLU(), last_activation=True)(x)

        assert a.shape == (10, 4, 12, 12)
        assert (a >= 0).all()

        a = Conv2dBlock(8, 4, kernel_size=3, depth=1, res_connection=True, activation=torch.nn.ReLU(), last_activation=True)(x)

        assert a.shape == (10, 4, 12, 12)
        assert not (a >= 0).all()

        a = Conv2dBlock(8, 4, kernel_size=3, depth=1, res_connection=False, activation=torch.nn.ReLU(), last_activation=False)(x)

        assert a.shape == (10, 4, 12, 12)
        assert not (a >= 0).all()

        x = torch.rand(10, 8, 12, 12, 8)
        a = Conv3dBlock(8, 4, kernel_size=3, depth=1, res_connection=False, activation=torch.nn.ReLU(), last_activation=True)(x)

        assert a.shape == (10, 4, 12, 12, 8)
        assert (a >= 0).all()


def test_unet_attention():
    from mlspm.modules import UNetAttentionConv

    x = torch.rand(10, 8, 12, 12)
    q = torch.rand(10, 5, 3, 3)
    act = torch.nn.ReLU()

    with torch.no_grad():
        attention = UNetAttentionConv(8, 5, 12, 3, attention_activation="softmax", conv_activation=act)
        x_attn, attn = attention(x, q)

    assert x_attn.shape == (10, 8, 12, 12)
    assert attn.shape == (10, 12, 12)
    assert (attn <= 1).all() and (attn >= 0).all()
    for a in attn:
        assert torch.allclose(a.sum(), torch.tensor(1.0))


def test_AttentionConvZ():
    from mlspm.modules import AttentionConvZ

    x = torch.rand(10, 8, 12, 12, 5)

    with torch.no_grad():
        x = AttentionConvZ(8, 3)(x)

    assert x.shape == (10, 8, 12, 12, 3)
