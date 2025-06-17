"""Configuration validation tests for MLX WebSocket Server."""

import asyncio

import pytest

from .test_helpers import RealServerTestContext


class TestConfigurationValidation:
    """Test configuration parameter validation and edge cases"""

    @pytest.mark.asyncio
    async def test_temperature_boundaries(self):
        """Test temperature parameter boundary validation"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test valid temperature values
            for temp in [0.0, 0.5, 1.0]:
                await client.send({"type": "config", "temperature": temp})
                await asyncio.sleep(0.1)
                assert ctx.server.config["temperature"] == temp

            # Test invalid values (should be clamped or rejected)
            await client.send({"type": "config", "temperature": -0.5})  # Negative
            await asyncio.sleep(0.1)
            # Should either clamp to 0.0 or keep previous value
            assert 0.0 <= ctx.server.config["temperature"] <= 1.0

            await client.send({"type": "config", "temperature": 2.0})  # Too high
            await asyncio.sleep(0.1)
            assert 0.0 <= ctx.server.config["temperature"] <= 1.0

            await client.close()

    @pytest.mark.asyncio
    async def test_max_tokens_validation(self):
        """Test maxOutputTokens parameter validation"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test valid values (maxOutputTokens is converted to maxTokens)
            for tokens in [10, 100, 1000]:
                await client.send({"type": "config", "maxOutputTokens": tokens})
                await asyncio.sleep(0.1)
                assert ctx.server.config["maxTokens"] == tokens  # Converted

            # Test edge cases
            await client.send({"type": "config", "maxOutputTokens": 0})  # Zero tokens
            await asyncio.sleep(0.1)
            # Should either reject or set to minimum
            assert ctx.server.config["maxTokens"] > 0  # Still converted

            await client.close()

    @pytest.mark.asyncio
    async def test_top_k_top_p_validation(self):
        """Test topK and topP parameter validation"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test topP boundaries (probability must be 0-1)
            for p in [0.0, 0.5, 0.9, 1.0]:
                await client.send({"type": "config", "topP": p})
                await asyncio.sleep(0.1)
                assert ctx.server.config["topP"] == p

            # Test invalid topP
            await client.send({"type": "config", "topP": 1.5})  # Invalid probability
            await asyncio.sleep(0.1)
            assert 0.0 <= ctx.server.config["topP"] <= 1.0

            # Test topK
            await client.send({"type": "config", "topK": 50})
            await asyncio.sleep(0.1)
            assert ctx.server.config["topK"] == 50

            await client.close()

    @pytest.mark.asyncio
    async def test_penalty_parameter_conversion(self):
        """Test conversion of OpenAI-style penalty parameters"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test presencePenalty conversion to repetitionPenalty
            await client.send({"type": "config", "presencePenalty": 0.5})
            await asyncio.sleep(0.1)
            # Should convert to repetitionPenalty
            assert "repetitionPenalty" in ctx.server.config

            # Test frequencyPenalty
            await client.send({"type": "config", "frequencyPenalty": 0.3})
            await asyncio.sleep(0.1)
            # Should also affect repetitionPenalty
            assert "repetitionPenalty" in ctx.server.config

            await client.close()

    @pytest.mark.asyncio
    async def test_multiple_config_updates(self):
        """Test multiple configuration parameters updated at once"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Update multiple parameters
            config_update = {
                "type": "config",
                "temperature": 0.8,
                "maxOutputTokens": 500,
                "topP": 0.95,
                "topK": 40,
                "seed": 12345,
            }

            await client.send(config_update)
            await asyncio.sleep(0.1)

            # Verify all were updated (note: maxOutputTokens is converted to maxTokens)
            assert ctx.server.config["temperature"] == 0.8
            assert ctx.server.config["maxTokens"] == 500  # Converted from maxOutputTokens
            assert ctx.server.config["topP"] == 0.95
            assert ctx.server.config["topK"] == 40
            assert ctx.server.config["seed"] == 12345

            await client.close()

    @pytest.mark.asyncio
    async def test_invalid_config_types(self):
        """Test handling of wrong types in config values"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Store original values
            orig_temp = ctx.server.config["temperature"]
            orig_tokens = ctx.server.config["maxTokens"]

            # Try invalid types
            await client.send(
                {
                    "type": "config",
                    "temperature": "not a number",
                    "maxOutputTokens": "also not a number",
                }
            )
            await asyncio.sleep(0.1)

            # Should keep original values on error
            assert ctx.server.config["temperature"] == orig_temp
            assert ctx.server.config["maxTokens"] == orig_tokens

            # Server should still be functional
            await client.send({"type": "text_input", "content": "Test after invalid config"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_unknown_config_parameters(self):
        """Test handling of unknown configuration parameters"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send unknown parameters
            await client.send(
                {
                    "type": "config",
                    "unknownParam": "value",
                    "anotherUnknown": 123,
                    "temperature": 0.7,  # Include one valid param
                }
            )
            await asyncio.sleep(0.1)

            # Valid param should be updated
            assert ctx.server.config["temperature"] == 0.7

            # Unknown params should be ignored (not crash)
            # Server should still work
            await client.send({"type": "text_input", "content": "Still working?"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_mlx_native_parameters(self):
        """Test MLX-native parameter configuration"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test MLX-native config update
            config_update = {
                "type": "config",
                "maxTokens": 300,  # MLX-native
                "repetitionPenalty": 1.2,  # MLX-native
                "repetitionContextSize": 30,  # MLX-native
                "seed": 42,
            }

            await client.send(config_update)
            await asyncio.sleep(0.1)

            # Verify MLX parameters were set directly
            assert ctx.server.config["maxTokens"] == 300
            assert ctx.server.config["repetitionPenalty"] == 1.2
            assert ctx.server.config["repetitionContextSize"] == 30
            assert ctx.server.config["seed"] == 42

            await client.close()
