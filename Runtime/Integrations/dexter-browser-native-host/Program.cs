using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

namespace DexterBrowserWorkspaceBridge
{
    internal sealed class Program : IDisposable
    {
        private const int MaxMessageBytes = 64 * 1024 * 1024;
        private readonly object _stdoutLock = new object();
        private readonly object _sessionLock = new object();
        private readonly object _jsonLock = new object();
        private readonly JavaScriptSerializer _json = new JavaScriptSerializer();
        private readonly Stream _stdin = Console.OpenStandardInput();
        private readonly Stream _stdout = Console.OpenStandardOutput();
        private readonly string _root;
        private readonly string _sessionsDir;
        private readonly string _requestsRoot;
        private readonly string _responsesDir;
        private readonly string _sessionId;
        private readonly string _sessionFile;
        private readonly string _requestDir;
        private readonly CancellationTokenSource _cancel = new CancellationTokenSource();
        private Thread _requestThread;
        private Thread _heartbeatThread;
        private volatile string _browserKind;
        private volatile string _extensionId;
        private bool _disposed;

        private Program()
        {
            string local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            _root = Path.Combine(local, "Dexter", "browser_bridge");
            _sessionsDir = Path.Combine(_root, "sessions");
            _requestsRoot = Path.Combine(_root, "requests");
            _responsesDir = Path.Combine(_root, "responses");
            _sessionId = "browser-" + Process.GetCurrentProcess().Id + "-" + Guid.NewGuid().ToString("N").Substring(0, 10);
            _sessionFile = Path.Combine(_sessionsDir, _sessionId + ".json");
            _requestDir = Path.Combine(_requestsRoot, _sessionId);

            Directory.CreateDirectory(_sessionsDir);
            Directory.CreateDirectory(_requestDir);
            Directory.CreateDirectory(_responsesDir);
            _json.MaxJsonLength = MaxMessageBytes;
        }

        private static int Main(string[] args)
        {
            using (var program = new Program())
            {
                try
                {
                    return program.Run();
                }
                catch
                {
                    // Native messaging stdout must contain protocol messages only.
                    // Exit quietly; Chrome surfaces connection failures to the extension.
                    return 1;
                }
            }
        }

        private int Run()
        {
            _requestThread = new Thread(RequestLoop) { IsBackground = true, Name = "DexterBrowserRequests" };
            _heartbeatThread = new Thread(HeartbeatLoop) { IsBackground = true, Name = "DexterBrowserHeartbeat" };
            _requestThread.Start();
            _heartbeatThread.Start();

            while (!_cancel.IsCancellationRequested)
            {
                Dictionary<string, object> message = ReadNativeMessage();
                if (message == null)
                    break;
                HandleExtensionMessage(message);
            }

            return 0;
        }

        private Dictionary<string, object> ReadNativeMessage()
        {
            byte[] lengthBytes = ReadExact(4);
            if (lengthBytes == null)
                return null;

            int length = lengthBytes[0]
                       | (lengthBytes[1] << 8)
                       | (lengthBytes[2] << 16)
                       | (lengthBytes[3] << 24);
            if (length <= 0 || length > MaxMessageBytes)
                throw new InvalidDataException("Invalid native message length.");

            byte[] payload = ReadExact(length);
            if (payload == null)
                throw new EndOfStreamException();
            string text = Encoding.UTF8.GetString(payload);
            return DeserializeMessage(text);
        }

        private byte[] ReadExact(int count)
        {
            byte[] buffer = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = _stdin.Read(buffer, offset, count - offset);
                if (read <= 0)
                {
                    if (offset == 0)
                        return null;
                    throw new EndOfStreamException();
                }
                offset += read;
            }
            return buffer;
        }

        private void WriteNativeMessage(object value)
        {
            string text = SerializeJson(value);
            byte[] payload = Encoding.UTF8.GetBytes(text);
            if (payload.Length > MaxMessageBytes)
                throw new InvalidDataException("Native message exceeded size limit.");

            byte[] prefix = new byte[4];
            int length = payload.Length;
            prefix[0] = (byte)(length & 0xFF);
            prefix[1] = (byte)((length >> 8) & 0xFF);
            prefix[2] = (byte)((length >> 16) & 0xFF);
            prefix[3] = (byte)((length >> 24) & 0xFF);

            lock (_stdoutLock)
            {
                _stdout.Write(prefix, 0, prefix.Length);
                _stdout.Write(payload, 0, payload.Length);
                _stdout.Flush();
            }
        }

        private void HandleExtensionMessage(Dictionary<string, object> message)
        {
            string type = StringValue(message, "type");
            if (type == "hello")
            {
                string kind = StringValue(message, "browser_kind");
                if (kind == "brave" || kind == "chrome" || kind == "edge")
                    _browserKind = kind;
                _extensionId = StringValue(message, "extension_id");
                WriteSessionRecord();
                return;
            }

            if (type != "response")
                return;

            string requestId = StringValue(message, "request_id");
            if (String.IsNullOrWhiteSpace(requestId) || !SafeId(requestId))
                return;

            string responsePath = Path.Combine(_responsesDir, requestId + ".json");
            AtomicWriteText(responsePath, SerializeJson(message));
        }

        private void RequestLoop()
        {
            while (!_cancel.IsCancellationRequested)
            {
                try
                {
                    if (!String.IsNullOrWhiteSpace(_browserKind) && Directory.Exists(_requestDir))
                    {
                        string[] files = Directory.GetFiles(_requestDir, "*.json");
                        Array.Sort(files, StringComparer.OrdinalIgnoreCase);
                        foreach (string file in files)
                        {
                            if (_cancel.IsCancellationRequested)
                                break;
                            ProcessRequestFile(file);
                        }
                    }
                }
                catch { }
                _cancel.Token.WaitHandle.WaitOne(45);
            }
        }

        private void ProcessRequestFile(string path)
        {
            string text;
            try
            {
                text = File.ReadAllText(path, Encoding.UTF8);
            }
            catch
            {
                return;
            }

            Dictionary<string, object> request;
            try
            {
                request = DeserializeMessage(text);
            }
            catch
            {
                TryDelete(path);
                return;
            }

            string requestId = StringValue(request, "request_id");
            string op = StringValue(request, "op");
            if (String.IsNullOrWhiteSpace(requestId) || !SafeId(requestId) || String.IsNullOrWhiteSpace(op))
            {
                TryDelete(path);
                return;
            }

            object payload;
            request.TryGetValue("payload", out payload);
            var outgoing = new Dictionary<string, object>
            {
                { "type", "request" },
                { "request_id", requestId },
                { "op", op },
                { "payload", payload ?? new Dictionary<string, object>() }
            };

            try
            {
                WriteNativeMessage(outgoing);
                TryDelete(path);
            }
            catch
            {
                // Keep the request on disk so a transient write problem can retry.
            }
        }

        private void HeartbeatLoop()
        {
            while (!_cancel.IsCancellationRequested)
            {
                try { WriteSessionRecord(); } catch { }
                _cancel.Token.WaitHandle.WaitOne(1000);
            }
        }

        private void WriteSessionRecord()
        {
            if (String.IsNullOrWhiteSpace(_browserKind))
                return;
            var record = new Dictionary<string, object>
            {
                { "session_id", _sessionId },
                { "browser_kind", _browserKind },
                { "extension_id", _extensionId ?? "" },
                { "pid", Process.GetCurrentProcess().Id },
                { "updated_unix", (DateTime.UtcNow - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds }
            };
            lock (_sessionLock)
            {
                AtomicWriteText(_sessionFile, SerializeJson(record));
            }
        }


        private string SerializeJson(object value)
        {
            lock (_jsonLock)
            {
                return _json.Serialize(value);
            }
        }

        private Dictionary<string, object> DeserializeMessage(string text)
        {
            lock (_jsonLock)
            {
                return _json.Deserialize<Dictionary<string, object>>(text);
            }
        }

        private static string StringValue(Dictionary<string, object> message, string key)
        {
            object value;
            if (!message.TryGetValue(key, out value) || value == null)
                return null;
            return Convert.ToString(value);
        }

        private static bool SafeId(string value)
        {
            foreach (char c in value)
            {
                if (!(Char.IsLetterOrDigit(c) || c == '_' || c == '-'))
                    return false;
            }
            return value.Length > 0 && value.Length <= 160;
        }

        private static void AtomicWriteText(string target, string text)
        {
            string directory = Path.GetDirectoryName(target);
            if (!String.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            string temp = target + "." + Process.GetCurrentProcess().Id + ".tmp";
            File.WriteAllText(temp, text, new UTF8Encoding(false));
            if (File.Exists(target))
                File.Delete(target);
            File.Move(temp, target);
        }

        private static void TryDelete(string path)
        {
            try { File.Delete(path); } catch { }
        }

        public void Dispose()
        {
            if (_disposed)
                return;
            _disposed = true;
            _cancel.Cancel();
            TryDelete(_sessionFile);
            try
            {
                if (Directory.Exists(_requestDir) && Directory.GetFileSystemEntries(_requestDir).Length == 0)
                    Directory.Delete(_requestDir);
            }
            catch { }
            _cancel.Dispose();
        }
    }
}
