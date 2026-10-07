# Native virtual camera — transport prepared

Milestone 9 defines and implements the producer side of the versioned shared-memory ABI.
The future C++ Media Foundation source must follow
[`docs/frame_transport_protocol.md`](../../docs/frame_transport_protocol.md) exactly. It
will consume completed frames and contain no tracking or face-rendering logic.

No Media Foundation source, camera registration, lifecycle tooling, or installer exists
yet. Friendly name: FaceLive Virtual Camera.
