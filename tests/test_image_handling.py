"""Image and frame handling tests for MLX WebSocket Server."""

import asyncio
import base64
import io

import pytest
from PIL import Image

from .test_helpers import RealServerTestContext


class TestImageHandling:
    """Test image processing and edge cases"""

    @pytest.mark.asyncio
    async def test_multiple_image_formats(self):
        """Test handling of different image formats"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test different formats
            formats = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "BMP": "bmp"}

            for fmt_name, fmt_ext in formats.items():
                # Skip WEBP if not supported
                if fmt_name == "WEBP":
                    try:
                        img = Image.new("RGB", (100, 100))
                        img.save(io.BytesIO(), format="WEBP")
                    except Exception:
                        continue

                # Create test image
                img = Image.new("RGB", (100, 100), color="blue")
                img_buffer = io.BytesIO()

                # Save in specific format
                save_kwargs = {"format": fmt_name}
                if fmt_name == "JPEG":
                    save_kwargs["quality"] = 95

                img.save(img_buffer, **save_kwargs)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

                # Send image
                await client.send(
                    {
                        "type": "image_input",
                        "image": f"data:image/{fmt_ext};base64,{img_base64}",
                        "prompt": f"What format is this {fmt_name} image?",
                    }
                )

                # Should process without error
                response = await client.receive_all()
                assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_large_image_resizing(self):
        """Test automatic resizing of large images"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Create large image (1024x1024, exceeds 768px limit)
            large_img = Image.new("RGB", (1024, 1024), color="red")
            img_buffer = io.BytesIO()
            large_img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Send large image
            await client.send(
                {
                    "type": "image_input",
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "Describe this large image",
                }
            )

            # Should handle and resize automatically
            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_invalid_base64_image(self):
        """Test handling of invalid base64 image data"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Send invalid base64
            await client.send(
                {
                    "type": "image_input",
                    "image": "data:image/png;base64,INVALID_BASE64_DATA!!!",
                    "prompt": "What's in this image?",
                }
            )

            # Should handle error gracefully
            # Send valid message after
            await client.send(
                {"type": "text_input", "content": "Can you still respond after the invalid image?"}
            )

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_missing_data_url_parts(self):
        """Test handling of malformed data URLs"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Test various malformed data URLs
            malformed_urls = [
                "not_a_data_url",
                "data:image/png",  # Missing base64 part
                "data:;base64,ABC123",  # Missing MIME type
                "data:image/png;base64",  # Missing data
                "image/png;base64,ABC123",  # Missing data: prefix
            ]

            for url in malformed_urls:
                await client.send(
                    {"type": "image_input", "image": url, "prompt": "Test malformed URL"}
                )

                # Brief pause
                await asyncio.sleep(0.1)

            # Server should still be responsive
            await client.send(
                {"type": "text_input", "content": "Still working after malformed URLs?"}
            )

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_video_frame_vs_image(self):
        """Test different frame types (video_frame, screen_frame)"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Create test image
            img = Image.new("RGB", (200, 200), color="green")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Test video_frame type
            await client.send(
                {
                    "type": "video_frame",
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "What's in this video frame?",
                }
            )

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            # Test screen_frame type
            await client.send(
                {
                    "type": "screen_frame",
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "What's on this screen?",
                }
            )

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    @pytest.mark.asyncio
    async def test_rapid_frame_sending(self):
        """Test handling of rapid frame sending (queue overflow)"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Create small test image
            img = Image.new("RGB", (50, 50), color="yellow")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Send many frames rapidly
            for i in range(20):
                await client.send(
                    {
                        "type": "video_frame",
                        "image": f"data:image/png;base64,{img_base64}",
                        "prompt": f"Frame {i}",
                    }
                )
                # No delay - simulate rapid sending

            # Give time for processing
            await asyncio.sleep(2.0)

            # Server should handle overflow gracefully
            await client.send({"type": "text_input", "content": "Did you handle all those frames?"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()

    @pytest.mark.asyncio
    async def test_empty_image_prompt(self):
        """Test image input with empty or missing prompt"""
        async with RealServerTestContext() as ctx:
            client = ctx.create_client()
            await client.connect()

            # Create test image
            img = Image.new("RGB", (100, 100), color="purple")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

            # Send image with empty prompt
            await client.send(
                {
                    "type": "image_input",
                    "image": f"data:image/png;base64,{img_base64}",
                    "prompt": "",
                }
            )

            # Should handle gracefully
            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            # Send image with missing prompt field
            await client.send(
                {
                    "type": "image_input",
                    "image": f"data:image/png;base64,{img_base64}",
                    # No prompt field
                }
            )

            # Should still work or handle error gracefully
            await asyncio.sleep(0.5)

            # Verify server still responsive
            await client.send({"type": "text_input", "content": "Still working?"})

            response = await client.receive_all()
            assert any(msg["type"] == "response_complete" for msg in response)

            await client.close()
