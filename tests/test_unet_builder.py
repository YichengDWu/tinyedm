from tinyedm.unet import get_decoder_out_channels, get_encoder_out_channels, get_skip_channels, get_skip_connections

def test_skip_channels():
    decoder_out_channels = get_decoder_out_channels()
    encoder_out_channels = get_encoder_out_channels()
    skip_connections = get_skip_connections()
    
    assert len(decoder_out_channels) == 21, "Expected 21 decoder out channels, got {len(decoder_out_channels)}"
    assert len(encoder_out_channels) == 15, "Expected 15 encoder out channels, got {len(encoder_out_channels)}"
    assert len(skip_connections) == 16, "Expected 16 skip connections, got {len(skip_connections)}"
    skip_channels = get_skip_channels(encoder_out_channels, decoder_out_channels, skip_connections)
    
    assert len(skip_channels) == 16, "Expected 16 skip channels, got {len(skip_channels)}"