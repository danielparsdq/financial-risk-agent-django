from __future__ import annotations

import json
import os
import inspect
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Callable, Dict, List
from django.conf import settings
from openai import OpenAI

# Configuración de logging para la consola
logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

def _setting(name: str, default: Any) -> Any:
    return os.getenv(name, str(default))


@dataclass(slots=True)
class BaseAgent:
    name: str
    system_prompt: str
    audit_rules: str | None = None
    model: str | None = None
    host: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    timeout: int | None = None
    tools: Dict[str, Callable] = field(default_factory=dict)
    tool_schemas: Dict[str, Dict] = field(default_factory=dict)
    client: OpenAI | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = self.model or _setting("LLM_MODEL", "gemma2:9b")
        self.host = self.host or _setting("LLM_HOST", "http://127.0.0.1:11434/v1")
        self.api_key = self.api_key or _setting("LLM_API_KEY", "ollama")
        self.temperature = float(self.temperature if self.temperature is not None else _setting("LLM_TEMPERATURE", 0.0))
        self.timeout = int(self.timeout if self.timeout is not None else _setting("LLM_TIMEOUT", 120))
        
        self.client = OpenAI(
            base_url=self.host,
            api_key=self.api_key,
            timeout=self.timeout
        )

    def add_tool(self, name: str, func: Callable, schema: Dict[str, Any] | None = None):
        """Registra una nueva herramienta que el agente puede usar."""
        self.tools[name] = func
        if schema:
            self.tool_schemas[name] = schema

    def _get_tools_definition(self) -> List[Dict[str, Any]]:
        """Genera la definición de herramientas para la API de Ollama/LLM."""
        definitions = []
        for name, func in self.tools.items():
            # Si hay un esquema manual, usarlo
            if name in self.tool_schemas:
                schema = self.tool_schemas[name]
                definitions.append({
                    "type": "function",
                    "function": schema
                })
                continue

            sig = inspect.signature(func)
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            for param_name, param in sig.parameters.items():
                if param_name == 'self': continue
                
                param_type = "string" # Default
                if param.annotation == int: param_type = "integer"
                elif param.annotation == float: param_type = "number"
                elif param.annotation == bool: param_type = "boolean"
                
                parameters["properties"][param_name] = {
                    "type": param_type,
                    "description": f"Parameter {param_name}"
                }
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)
            
            definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": func.__doc__ or f"Tool {name}",
                    "parameters": parameters
                }
            })
        return definitions

    def _call_tool(self, tool_call: Any) -> str:
        """Ejecuta una herramienta solicitada por el modelo."""
        # Manejar tanto dict (Ollama) como objetos (OpenAI)
        if hasattr(tool_call, 'function'):
            name = tool_call.function.name
            arguments_raw = tool_call.function.arguments
        else:
            function_info = tool_call.get("function", {})
            name = function_info.get("name")
            arguments_raw = function_info.get("arguments", {})

        if isinstance(arguments_raw, str):
            try:
                args = json.loads(arguments_raw)
            except json.JSONDecodeError:
                args = {}
        else:
            args = arguments_raw
        
        if name in self.tools:
            logger.info(f"🛠️ [HERRAMIENTA]: Usando '{name}' con argumentos: {args}")
            try:
                result = self.tools[name](**args)
                logger.info(f"✅ [HERRAMIENTA]: '{name}' ejecutada con éxito.")
                logger.info(f"📦 [HERRAMIENTA]: Resultado: {result}")
                return str(result)
            except Exception as e:
                logger.error(f"❌ [HERRAMIENTA]: Error en '{name}': {str(e)}")
                return f"Error executing tool {name}: {str(e)}"
        logger.warning(f"⚠️ [HERRAMIENTA]: No se encontró '{name}'")
        return f"Tool {name} not found"

    def chat(self, prompt: str, context: str | None = None, skip_audit: bool = False) -> str:
        messages = self._build_messages([{ "role": "user", "content": prompt }], context=context)
        
        while True:
            logger.info(f"🧠 [PENSANDO]: Consultando al modelo {self.model}...")
            start_time = time.time()
            tools_def = self._get_tools_definition() if self.tools else None
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools_def if tools_def else None
                )
                exec_time = int((time.time() - start_time) * 1000)
                message = response.choices[0].message
                
                # Si el modelo no pide herramientas, procesamos el final del chat
                if not message.tool_calls:
                    content = (message.content or "").strip()
                    logger.info(f"✍️ [RESPONDIENDO]: El modelo generó una respuesta ({exec_time}ms)")
                    
                    # --- AUDITORÍA DE CUMPLIMIENTO ---
                    error_message = ""
                    
                    if skip_audit:
                        status = "success"
                        audit_data = {}
                    else:
                        logger.info(f"🔍 [AUDITANDO]: Verificando cumplimiento normativo...")
                        audit_data = self.audit_compliance(prompt, content)
                        
                        if not audit_data.get("compliance_passed", True):
                            failure_msg = audit_data.get("failure_reason", "No cumple con las normas de seguridad del servicio público.")
                            logger.warning(f"🚫 [BLOQUEADO]: La respuesta no pasó la auditoría. Motivo: {failure_msg}")
                            error_message = failure_msg
                            content = (
                                f"⚠️ Error message: {failure_msg}\n\n"
                                f"La respuesta generada ha sido bloqueada por no cumplir con las normativas de IA (XIA Europa) "
                                f"o Protección de Datos (LOPDP Ecuador)."
                            )
                            status = "blocked"
                        else:
                            status = "success"

                    # Logueamos la transacción con los datos de auditoría
                    self._log_transaction(
                        path="/v1/chat/completions",
                        request_payload={"messages": messages, "model": self.model},
                        response_payload=response.model_dump(),
                        status=status,
                        error_message=error_message,
                        audit_data=audit_data,
                        execution_time_ms=exec_time
                    )
                    
                    return content
                
                # Si pide herramientas, las ejecutamos y añadimos los resultados a la conversación
                messages.append(message)
                for tool_call in message.tool_calls:
                    result = self._call_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
            except Exception as e:
                exec_time = int((time.time() - start_time) * 1000)
                logger.error(f"❌ [ERROR]: Error en comunicación con LLM: {str(e)}")
                self._log_transaction(
                    path="/v1/chat/completions",
                    request_payload={"messages": messages, "model": self.model},
                    response_payload=None,
                    status="error",
                    error_message=str(e),
                    execution_time_ms=exec_time
                )
                raise ConnectionError(f"Error reaching LLM: {str(e)}") from e
            # El bucle continuará para que el modelo procese los resultados de las herramientas

    def generate(self, prompt: str, context: str | None = None, skip_audit: bool = False) -> str:
        return self.chat(prompt, context, skip_audit=skip_audit)

    def chat_messages(self, conversation: list[dict[str, str]], context: str | None = None, skip_audit: bool = False) -> str:
        messages = self._build_messages(conversation, context=context)
        last_user_msg = next((m.get("content") for m in reversed(conversation) if m.get("role") == "user"), "")

        while True:
            logger.info(f"🧠 [PENSANDO]: Procesando historial con {self.model}...")
            start_time = time.time()
            tools_def = self._get_tools_definition() if self.tools else None
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools_def if tools_def else None
                )
                exec_time = int((time.time() - start_time) * 1000)
                message = response.choices[0].message
                
                if not message.tool_calls:
                    content = (message.content or "").strip()
                    logger.info(f"✍️ [RESPONDIENDO]: Respuesta generada en {exec_time}ms")
                    
                    # --- AUDITORÍA DE CUMPLIMIENTO ---
                    error_message = ""
                    
                    if skip_audit:
                        status = "success"
                        audit_data = {}
                    else:
                        logger.info(f"🔍 [AUDITANDO]: Evaluando respuesta contra normativas...")
                        audit_data = self.audit_compliance(last_user_msg, content)
                        
                        if not audit_data.get("compliance_passed", True):
                            failure_msg = audit_data.get("failure_reason", "No cumple con las normas de seguridad.")
                            logger.warning(f"🚫 [BLOQUEADO]: Auditoría fallida. Motivo: {failure_msg}")
                            error_message = failure_msg
                            content = f"⚠️ Error message: {failure_msg}"
                            status = "blocked"
                        else:
                            status = "success"

                    self._log_transaction(
                        path="/v1/chat/completions",
                        request_payload={"messages": messages, "model": self.model},
                        response_payload=response.model_dump(),
                        status=status,
                        error_message=error_message,
                        audit_data=audit_data,
                        execution_time_ms=exec_time
                    )
                    
                    return content
                
                messages.append(message)
                for tool_call in message.tool_calls:
                    result = self._call_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id
                    })
            except Exception as e:
                exec_time = int((time.time() - start_time) * 1000)
                logger.error(f"❌ [ERROR]: Error en comunicación con LLM: {str(e)}")
                self._log_transaction(
                    path="/v1/chat/completions",
                    request_payload={"messages": messages, "model": self.model},
                    response_payload=None,
                    status="error",
                    error_message=str(e),
                    execution_time_ms=exec_time
                )
                raise ConnectionError(f"Error reaching LLM: {str(e)}") from e

    def stream_chat(self, prompt: str, context: str | None = None) -> Iterator[str]:
        yield from self.stream_chat_messages([{ "role": "user", "content": prompt }], context=context)

    def stream_chat_messages(
        self,
        conversation: list[dict[str, str]],
        context: str | None = None,
        skip_audit: bool = False,
    ) -> Iterator[str]:
        messages = self._build_messages(conversation, context=context)
        last_user_msg = next((m.get("content") for m in reversed(conversation) if m.get("role") == "user"), "")
        start_time = time.time()
        
        while True:
            logger.info(f"🧠 [PENSANDO/STREAM]: Iniciando flujo con {self.model}...")
            tools_def = self._get_tools_definition() if self.tools else None
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    tools=tools_def if tools_def else None,
                    stream=True
                )
                
                full_content = ""
                tool_calls = []
                
                for chunk in response:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    
                    # Manejar contenido de texto
                    if delta.content:
                        piece = delta.content
                        full_content += piece
                        yield piece
                        
                    # Manejar llamadas a herramientas (Ollama/OpenAI stream format)
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            if len(tool_calls) <= tc_delta.index:
                                tool_calls.append({
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": tc_delta.function.name, "arguments": ""}
                                })
                            if tc_delta.function.arguments:
                                tool_calls[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments

                if not tool_calls:
                    exec_time = int((time.time() - start_time) * 1000)
                    logger.info(f"✍️ [RESPONDIENDO]: Stream finalizado en {exec_time}ms")
                    
                    audit_data = {}
                    if not skip_audit:
                        audit_data = self.audit_compliance(last_user_msg, full_content)
                    
                    self._log_transaction(
                        path="/v1/chat/completions?stream=true",
                        request_payload={"messages": messages, "model": self.model},
                        response_payload={"content": full_content},
                        status="success" if audit_data.get("compliance_passed", True) else "blocked",
                        audit_data=audit_data,
                        execution_time_ms=exec_time
                    )
                    return # Final normal del stream

                # Si hay herramientas, procesarlas y continuar el bucle
                messages.append({
                    "role": "assistant",
                    "content": full_content,
                    "tool_calls": tool_calls
                })
                
                for tc in tool_calls:
                    # Convertir tc a un objeto compatible con _call_tool
                    class ToolCall:
                        def __init__(self, d):
                            self.id = d['id']
                            class Function:
                                def __init__(self, f):
                                    self.name = f['name']
                                    self.arguments = f['arguments']
                            self.function = Function(d['function'])
                    
                    result = self._call_tool(ToolCall(tc))
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc["id"]
                    })
                # El bucle continúa para enviar los resultados de las herramientas de vuelta al modelo

            except Exception as e:
                exec_time = int((time.time() - start_time) * 1000)
                logger.error(f"❌ [ERROR/STREAM]: {str(e)}")
                self._log_transaction(
                    path="/v1/chat/completions?stream=true",
                    request_payload={"messages": messages, "model": self.model},
                    response_payload=None,
                    status="error",
                    error_message=str(e),
                    execution_time_ms=exec_time
                )
                raise ConnectionError(f"Error in stream: {e}") from e

    def _build_messages(self, conversation: list[dict[str, str]], context: str | None = None) -> list[dict[str, str]]:
        """Construye la lista de mensajes incluyendo el prompt del sistema y el contexto."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "system", "content": f"Contexto adicional:\n{context}"})
        messages.extend(conversation)
        return messages

    ### Auditoría de cumplimiento de normativas XIA y LOPDP
    def audit_compliance(self, prompt: str, response_text: str) -> dict[str, Any]:
        """Analiza la interacción y verifica el cumplimiento de normativas EU AI Act y LOPDP (EC)."""
        logger.info("⚖️ [AUDITORÍA]: Iniciando análisis de cumplimiento normativo (EU/EC)...")
        
        audit_prompt = f"""Eres un auditor legal experto en la Ley de Inteligencia Artificial de la UE (AI Act) y la Ley Orgánica de Protección de Datos Personales (LOPDP) de Ecuador.
        Tu tarea es evaluar si la RESPUESTA DEL AGENTE cumple con las normativas frente a la consulta del usuario.

        {self.audit_rules}
        
        Interacción a evaluar:
        Pregunta Ciudadano: "{prompt}"
        Respuesta Agente: "{response_text}"

        Responde SOLO con un JSON válido usando exactamente esta estructura:
        {{
            "ai_risk_level": "low"|"medium"|"high"|"unacceptable", 
            "human_oversight_needed": true|false,
            "transparency_confirmed": true|false,
            "legal_basis": "consent"|"pre_contractual"|"legitimate_interest"|"none_yet",
            "pii_requested_by_agent": true|false,
            "data_minimization_applied": true|false,
            "compliance_passed": true|false,
            "failure_reason": "Si compliance_passed es false, explica qué regla infringió EL AGENTE. Si es true, deja este campo en blanco."
        }}
        """
        
        audit_model = getattr(settings, 'AUDIT_LLM_MODEL', self.model)

        try:
            response = self.client.chat.completions.create(
                model=audit_model,
                messages=[{"role": "user", "content": audit_prompt}],
                response_format={"type": "json_object"}
            )
            audit_content = response.choices[0].message.content
            logger.info("✅ [AUDITORÍA]: Análisis completado exitosamente.")
            return json.loads(audit_content)
        except Exception as e:
            logger.error(f"⚠️ [AUDITORÍA]: Error en proceso de auditoría: {e}")
            return {
                "compliance_passed": True, 
                "ai_risk_level": "medium", 
                "failure_reason": f"Error del auditor: {str(e)}. Omitido."
            }

    def _log_transaction(
        self,
        path: str,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any] | None,
        status: str,
        error_message: str = "",
        audit_data: dict[str, Any] | None = None,
        execution_time_ms: int = 0
    ) -> None:
        try:
            from audit.models import AgentTransactionLog

            log_params = {
                "agent_name": self.name,
                "endpoint": path,
                "request_payload": request_payload,
                "response_payload": response_payload,
                "status": status,
                "error_message": error_message,
                "model_version": self.model or "unknown",
                "execution_time_ms": execution_time_ms,
            }

            if audit_data:
                # Mapear campos del audit_data a los campos del modelo
                # Asegurándonos de que coincidan con la estructura del modelo
                fields_to_map = [
                    'ai_risk_level', 'human_oversight', 'transparency_confirmed',
                    'token_usage', 'legal_basis', 'pii_detected',
                    'data_minimization_applied', 'anonymized_payload',
                    'consent_record_id', 'data_retention_days'
                ]
                for f in fields_to_map:
                    if f in audit_data:
                        log_params[f] = audit_data[f]

            AgentTransactionLog.objects.create(**log_params)
        except Exception as e:
            # Logging should never break agent execution.
            print(f"Error logging transaction: {e}")
            return